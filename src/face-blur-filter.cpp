// SPDX-License-Identifier: GPL-2.0-or-later
//
// Face Blur Tracker — видеофильтр OBS: находит лица (YuNet / OpenCV),
// сопровождает их трекером и размывает на GPU.
//
// Схема работы:
//   render-поток OBS  -> уменьшенная копия кадра -> staging surface
//   поток детектора   -> YuNet -> трекер (сглаживание, удержание рамки)
//   render-поток OBS  <- позиции лиц -> шейдер face_blur.effect

#define NOMINMAX
#include <obs-module.h>
#include <graphics/graphics.h>
#include <graphics/vec4.h>
#include <util/platform.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "face-tracker.hpp"
#include "yunet-detector.hpp"

#define T_(s) obs_module_text(s)

#define S_MODE "mode"
#define S_STRENGTH "strength"
#define S_PADDING "padding"
#define S_FEATHER "feather"
#define S_COLOR "solid_color"
#define S_CONF "confidence"
#define S_HOLD "hold_ms"
#define S_MAXFACES "max_faces"
#define S_DETW "detect_width"
#define S_FPS "detect_fps"
#define S_SMOOTH "smoothing"
#define S_DEBUG "debug_boxes"

static constexpr int MAX_FACES = 16; // должно совпадать с faces[16] в шейдере
static const char *MODEL_FILE = "models/face_detection_yunet.onnx";

static double now_sec()
{
	return (double)os_gettime_ns() / 1e9;
}

struct face_blur_data {
	obs_source_t *context = nullptr;

	// --- графика (только в графическом потоке) ---
	gs_effect_t *effect = nullptr;
	gs_eparam_t *p_faces = nullptr, *p_count = nullptr, *p_mode = nullptr, *p_strength = nullptr,
		    *p_feather = nullptr, *p_aspect = nullptr, *p_texel = nullptr, *p_color = nullptr,
		    *p_debug = nullptr;
	gs_texrender_t *det_render = nullptr;
	gs_stagesurf_t *stage = nullptr;
	uint32_t det_w = 0, det_h = 0;
	bool stage_queued = false;
	uint64_t last_capture_ns = 0;

	// --- настройки ---
	int mode = 0;
	float strength = 0.6f;
	float padding = 0.4f;
	float feather = 0.2f;
	uint32_t color = 0xFF000000;
	bool debug = false;
	uint32_t detect_width = 480;
	uint64_t detect_interval_ns = 33333333ULL;
	std::atomic<float> conf{0.7f};

	// --- поток детектора ---
	std::thread worker;
	std::mutex mtx;
	std::condition_variable cv;
	bool quit = false;
	bool has_frame = false;
	cv::Mat frame; // RGBA
	std::string model_path;

	// --- трекер (общий для потока детектора и render-потока) ---
	std::mutex track_mtx;
	FaceTracker tracker;
	std::atomic<bool> model_ok{true};
};

// ---------------------------------------------------------------------------
// Поток детектора
// ---------------------------------------------------------------------------

static void worker_main(face_blur_data *f)
{
	YuNetDetector detector;
	cv::Mat bgr;
	bool loaded = false;
	bool create_failed = false;

	for (;;) {
		cv::Mat local;
		{
			std::unique_lock<std::mutex> lk(f->mtx);
			f->cv.wait(lk, [f] { return f->quit || f->has_frame; });
			if (f->quit)
				break;
			local = std::move(f->frame);
			f->has_frame = false;
		}
		if (local.empty() || create_failed)
			continue;

		std::vector<FaceBox> dets;
		try {
			cv::cvtColor(local, bgr, cv::COLOR_RGBA2BGR);

			if (!loaded) {
				if (!detector.load(f->model_path))
					CV_Error(cv::Error::StsError, "cannot load model");
				loaded = true;
			}
			detector.detect(bgr, f->conf.load(), 0.3f, dets);
			dets.erase(std::remove_if(dets.begin(), dets.end(),
						  [](const FaceBox &b) { return b.w <= 0.0f || b.h <= 0.0f; }),
				   dets.end());
		} catch (const cv::Exception &e) {
			blog(LOG_ERROR, "[obs-face-blur] detector error: %s", e.what());
			if (!loaded) {
				create_failed = true;
				f->model_ok = false;
			}
			continue;
		}

		std::lock_guard<std::mutex> lk(f->track_mtx);
		f->tracker.update(std::move(dets), now_sec());
	}
}

// ---------------------------------------------------------------------------
// Захват кадра для детекции (в графическом потоке)
// ---------------------------------------------------------------------------

static void free_capture_resources(face_blur_data *f)
{
	if (f->stage) {
		gs_stagesurface_destroy(f->stage);
		f->stage = nullptr;
	}
	if (f->det_render) {
		gs_texrender_destroy(f->det_render);
		f->det_render = nullptr;
	}
	f->stage_queued = false;
	f->det_w = f->det_h = 0;
}

static void capture_for_detection(face_blur_data *f, obs_source_t *target, uint32_t bw, uint32_t bh)
{
	uint32_t dw = std::min(f->detect_width, bw);
	uint32_t dh = std::max<uint32_t>(1, (uint32_t)((uint64_t)bh * dw / bw));
	dw &= ~1u;
	dh &= ~1u;
	if (dw < 32 || dh < 32)
		return;

	if (!f->det_render || !f->stage || dw != f->det_w || dh != f->det_h) {
		free_capture_resources(f);
		f->det_render = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
		f->stage = gs_stagesurface_create(dw, dh, GS_RGBA);
		f->det_w = dw;
		f->det_h = dh;
	}
	if (!f->det_render || !f->stage)
		return;

	// 1) забираем кадр, поставленный в очередь на прошлом проходе (без стопора GPU)
	if (f->stage_queued) {
		f->stage_queued = false;
		uint8_t *data = nullptr;
		uint32_t linesize = 0;
		if (gs_stagesurface_map(f->stage, &data, &linesize)) {
			cv::Mat m((int)dh, (int)dw, CV_8UC4);
			for (uint32_t y = 0; y < dh; y++)
				std::memcpy(m.ptr((int)y), data + (size_t)y * linesize, (size_t)dw * 4);
			gs_stagesurface_unmap(f->stage);
			{
				std::lock_guard<std::mutex> lk(f->mtx);
				f->frame = std::move(m); // старый необработанный кадр отбрасываем
				f->has_frame = true;
			}
			f->cv.notify_one();
		}
	}

	// 2) не чаще заданной частоты детекции
	uint64_t now = os_gettime_ns();
	if (now - f->last_capture_ns < f->detect_interval_ns)
		return;
	f->last_capture_ns = now;

	// 3) рисуем источник в уменьшенную текстуру и копируем в staging
	gs_texrender_reset(f->det_render);
	if (gs_texrender_begin(f->det_render, dw, dh)) {
		struct vec4 clear;
		vec4_zero(&clear);
		gs_clear(GS_CLEAR_COLOR, &clear, 0.0f, 0);
		gs_ortho(0.0f, (float)bw, 0.0f, (float)bh, -100.0f, 100.0f);

		gs_blend_state_push();
		gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
		obs_source_video_render(target);
		gs_blend_state_pop();

		gs_texrender_end(f->det_render);

		gs_texture_t *tex = gs_texrender_get_texture(f->det_render);
		if (tex) {
			gs_stage_texture(f->stage, tex);
			f->stage_queued = true;
		}
	}
}

// ---------------------------------------------------------------------------
// Жизненный цикл фильтра
// ---------------------------------------------------------------------------

static const char *fb_get_name(void *)
{
	return T_("FB.Name");
}

static void fb_defaults(obs_data_t *s)
{
	obs_data_set_default_int(s, S_MODE, 0);
	obs_data_set_default_int(s, S_STRENGTH, 70);
	obs_data_set_default_int(s, S_PADDING, 40);
	obs_data_set_default_int(s, S_FEATHER, 25);
	obs_data_set_default_int(s, S_COLOR, 0xFF000000);
	obs_data_set_default_int(s, S_CONF, 70);
	obs_data_set_default_int(s, S_HOLD, 500);
	obs_data_set_default_int(s, S_MAXFACES, 8);
	obs_data_set_default_int(s, S_DETW, 480);
	obs_data_set_default_int(s, S_FPS, 30);
	obs_data_set_default_int(s, S_SMOOTH, 60);
	obs_data_set_default_bool(s, S_DEBUG, false);
}

static void fb_update(void *data, obs_data_t *s)
{
	auto *f = static_cast<face_blur_data *>(data);

	f->mode = (int)obs_data_get_int(s, S_MODE);
	f->strength = (float)obs_data_get_int(s, S_STRENGTH) / 100.0f;
	f->padding = (float)obs_data_get_int(s, S_PADDING) / 100.0f;
	f->feather = (float)obs_data_get_int(s, S_FEATHER) / 100.0f;
	f->color = (uint32_t)obs_data_get_int(s, S_COLOR);
	f->debug = obs_data_get_bool(s, S_DEBUG);
	f->detect_width = (uint32_t)std::max<long long>(160, obs_data_get_int(s, S_DETW));
	f->detect_interval_ns = 1000000000ULL / (uint64_t)std::max<long long>(1, obs_data_get_int(s, S_FPS));
	f->conf = (float)obs_data_get_int(s, S_CONF) / 100.0f;

	int max_faces = std::min<int>(MAX_FACES, (int)obs_data_get_int(s, S_MAXFACES));
	double hold = (double)obs_data_get_int(s, S_HOLD) / 1000.0;
	float smooth = (float)obs_data_get_int(s, S_SMOOTH) / 100.0f;

	std::lock_guard<std::mutex> lk(f->track_mtx);
	f->tracker.set_params(smooth, hold, max_faces);
}

static void *fb_create(obs_data_t *settings, obs_source_t *source)
{
	auto *f = new face_blur_data();
	f->context = source;

	char *effect_path = obs_module_file("effects/face_blur.effect");
	obs_enter_graphics();
	if (effect_path)
		f->effect = gs_effect_create_from_file(effect_path, nullptr);
	if (f->effect) {
		f->p_faces = gs_effect_get_param_by_name(f->effect, "faces");
		f->p_count = gs_effect_get_param_by_name(f->effect, "face_count");
		f->p_mode = gs_effect_get_param_by_name(f->effect, "mode");
		f->p_strength = gs_effect_get_param_by_name(f->effect, "strength");
		f->p_feather = gs_effect_get_param_by_name(f->effect, "feather");
		f->p_aspect = gs_effect_get_param_by_name(f->effect, "aspect");
		f->p_texel = gs_effect_get_param_by_name(f->effect, "texel");
		f->p_color = gs_effect_get_param_by_name(f->effect, "solid_color");
		f->p_debug = gs_effect_get_param_by_name(f->effect, "debug_boxes");
	}
	obs_leave_graphics();
	bfree(effect_path);

	if (!f->effect) {
		blog(LOG_ERROR, "[obs-face-blur] cannot load effects/face_blur.effect");
		delete f;
		return nullptr;
	}

	char *model = obs_module_file(MODEL_FILE);
	if (model) {
		f->model_path = model;
		bfree(model);
	}
	if (f->model_path.empty() || !os_file_exists(f->model_path.c_str())) {
		blog(LOG_ERROR, "[obs-face-blur] model file not found: data/%s (see README)", MODEL_FILE);
		f->model_ok = false;
	}

	fb_update(f, settings);
	if (f->model_ok)
		f->worker = std::thread(worker_main, f);
	return f;
}

static void fb_destroy(void *data)
{
	auto *f = static_cast<face_blur_data *>(data);

	if (f->worker.joinable()) {
		{
			std::lock_guard<std::mutex> lk(f->mtx);
			f->quit = true;
		}
		f->cv.notify_all();
		f->worker.join();
	}

	obs_enter_graphics();
	free_capture_resources(f);
	if (f->effect)
		gs_effect_destroy(f->effect);
	obs_leave_graphics();

	delete f;
}

// ---------------------------------------------------------------------------
// Отрисовка
// ---------------------------------------------------------------------------

static void fb_video_render(void *data, gs_effect_t *)
{
	auto *f = static_cast<face_blur_data *>(data);

	obs_source_t *target = obs_filter_get_target(f->context);
	obs_source_t *parent = obs_filter_get_parent(f->context);
	if (!target || !parent || !f->effect || !f->model_ok) {
		obs_source_skip_video_filter(f->context);
		return;
	}

	const uint32_t w = obs_source_get_base_width(target);
	const uint32_t h = obs_source_get_base_height(target);
	if (!w || !h) {
		obs_source_skip_video_filter(f->context);
		return;
	}

	capture_for_detection(f, target, w, h);

	// позиции лиц на текущий момент
	std::vector<RenderFace> faces;
	{
		std::lock_guard<std::mutex> lk(f->track_mtx);
		f->tracker.snapshot(now_sec(), faces);
	}
	const int n = (int)std::min<size_t>(faces.size(), MAX_FACES);

	if (n == 0 && !f->debug) {
		obs_source_skip_video_filter(f->context); // лиц нет — рисуем как есть
		return;
	}

	float rects[MAX_FACES * 4] = {0};
	for (int i = 0; i < n; i++) {
		const RenderFace &r = faces[(size_t)i];
		float fw = r.w * (1.0f + f->padding);
		float fh = r.h * (1.0f + f->padding * 1.1f);
		float cx = r.cx;
		float cy = r.cy - r.h * 0.04f; // чуть выше: захватываем лоб
		rects[i * 4 + 0] = cx - fw * 0.5f;
		rects[i * 4 + 1] = cy - fh * 0.5f;
		rects[i * 4 + 2] = fw;
		rects[i * 4 + 3] = fh;
	}

	if (!obs_source_process_filter_begin(f->context, GS_RGBA, OBS_ALLOW_DIRECT_RENDERING))
		return;

	gs_effect_set_val(f->p_faces, rects, sizeof(rects));
	gs_effect_set_int(f->p_count, n);
	gs_effect_set_int(f->p_mode, f->mode);
	gs_effect_set_float(f->p_strength, f->strength);
	gs_effect_set_float(f->p_feather, f->feather);
	gs_effect_set_float(f->p_aspect, (float)w / (float)h);
	struct vec2 texel;
	vec2_set(&texel, 1.0f / (float)w, 1.0f / (float)h);
	gs_effect_set_vec2(f->p_texel, &texel);
	struct vec4 col;
	vec4_from_rgba(&col, f->color);
	gs_effect_set_vec4(f->p_color, &col);
	gs_effect_set_int(f->p_debug, f->debug ? 1 : 0);

	obs_source_process_filter_end(f->context, f->effect, w, h);
}

// ---------------------------------------------------------------------------
// Свойства
// ---------------------------------------------------------------------------

static bool fb_mode_modified(obs_properties_t *props, obs_property_t *, obs_data_t *s)
{
	bool solid = obs_data_get_int(s, S_MODE) == 2;
	obs_property_set_visible(obs_properties_get(props, S_COLOR), solid);
	obs_property_set_visible(obs_properties_get(props, S_STRENGTH), !solid);
	return true;
}

static obs_properties_t *fb_properties(void *data)
{
	auto *f = static_cast<face_blur_data *>(data);
	obs_properties_t *p = obs_properties_create();

	if (f && !f->model_ok) {
		obs_properties_add_text(p, "model_warn", T_("FB.ModelMissing"), OBS_TEXT_INFO);
	}

	obs_property_t *mode = obs_properties_add_list(p, S_MODE, T_("FB.Mode"), OBS_COMBO_TYPE_LIST,
						       OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(mode, T_("FB.Mode.Blur"), 0);
	obs_property_list_add_int(mode, T_("FB.Mode.Pixelate"), 1);
	obs_property_list_add_int(mode, T_("FB.Mode.Solid"), 2);
	obs_property_set_modified_callback(mode, fb_mode_modified);

	obs_properties_add_int_slider(p, S_STRENGTH, T_("FB.Strength"), 1, 100, 1);
	obs_properties_add_color(p, S_COLOR, T_("FB.Color"));
	obs_properties_add_int_slider(p, S_PADDING, T_("FB.Padding"), 0, 150, 1);
	obs_properties_add_int_slider(p, S_FEATHER, T_("FB.Feather"), 0, 100, 1);

	obs_properties_add_int_slider(p, S_CONF, T_("FB.Confidence"), 30, 99, 1);
	obs_properties_add_int_slider(p, S_MAXFACES, T_("FB.MaxFaces"), 1, MAX_FACES, 1);
	obs_properties_add_int_slider(p, S_HOLD, T_("FB.Hold"), 0, 3000, 50);
	obs_properties_add_int_slider(p, S_SMOOTH, T_("FB.Smoothing"), 5, 100, 1);

	obs_property_t *dw = obs_properties_add_list(p, S_DETW, T_("FB.DetectWidth"), OBS_COMBO_TYPE_LIST,
						     OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(dw, "320", 320);
	obs_property_list_add_int(dw, "480", 480);
	obs_property_list_add_int(dw, "640", 640);
	obs_property_list_add_int(dw, "960", 960);

	obs_properties_add_int_slider(p, S_FPS, T_("FB.DetectFps"), 5, 60, 1);
	obs_properties_add_bool(p, S_DEBUG, T_("FB.Debug"));
	return p;
}

// ---------------------------------------------------------------------------

extern "C" void register_face_blur_filter(void)
{
	static struct obs_source_info info = {};
	info.id = "face_blur_tracker";
	info.type = OBS_SOURCE_TYPE_FILTER;
	info.output_flags = OBS_SOURCE_VIDEO;
	info.get_name = fb_get_name;
	info.create = fb_create;
	info.destroy = fb_destroy;
	info.update = fb_update;
	info.get_defaults = fb_defaults;
	info.get_properties = fb_properties;
	info.video_render = fb_video_render;
	obs_register_source(&info);
}
