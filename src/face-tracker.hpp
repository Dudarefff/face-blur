// SPDX-License-Identifier: GPL-2.0-or-later
// Простой трекер лиц: сопоставление по IoU + alpha-beta фильтр.
// Не зависит ни от OBS, ни от OpenCV — можно тестировать отдельно.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

struct FaceBox {
	float x, y, w, h; // нормализованные координаты (0..1), левый верхний угол
};

struct RenderFace {
	float cx, cy, w, h; // центр и размер, нормализованные
};

class FaceTracker {
public:
	// smoothing: 0.05..1 (больше = быстрее реагирует, но дрожит)
	// hold_sec:  сколько секунд держать рамку после потери лица
	void set_params(float smoothing, double hold_sec, int max_faces)
	{
		smoothing_ = std::min(1.0f, std::max(0.05f, smoothing));
		hold_sec_ = std::max(0.0, hold_sec);
		max_faces_ = std::max(1, max_faces);
	}

	void clear() { tracks_.clear(); }

	void update(std::vector<FaceBox> dets, double now)
	{
		// оставляем самые крупные лица, если их больше лимита
		std::sort(dets.begin(), dets.end(), [](const FaceBox &a, const FaceBox &b) {
			return a.w * a.h > b.w * b.h;
		});
		if ((int)dets.size() > max_faces_)
			dets.resize(max_faces_);

		struct Pair {
			float score;
			size_t t, d;
		};
		std::vector<Pair> pairs;
		for (size_t ti = 0; ti < tracks_.size(); ti++) {
			for (size_t di = 0; di < dets.size(); di++) {
				float s = match_score(tracks_[ti], dets[di]);
				if (s > 0.0f)
					pairs.push_back({s, ti, di});
			}
		}
		std::sort(pairs.begin(), pairs.end(),
			  [](const Pair &a, const Pair &b) { return a.score > b.score; });

		std::vector<char> t_used(tracks_.size(), 0), d_used(dets.size(), 0);
		for (const Pair &p : pairs) {
			if (t_used[p.t] || d_used[p.d])
				continue;
			t_used[p.t] = d_used[p.d] = 1;
			correct(tracks_[p.t], dets[p.d], now);
		}

		// потерянные треки удаляем по таймауту
		for (size_t i = tracks_.size(); i-- > 0;) {
			if (!t_used[i] && now - tracks_[i].last_seen > hold_sec_)
				tracks_.erase(tracks_.begin() + (long)i);
		}
		// t_used больше не соответствует индексам — новые треки добавляем после
		for (size_t di = 0; di < dets.size(); di++) {
			if (d_used[di])
				continue;
			if ((int)tracks_.size() >= max_faces_ * 2)
				break;
			Track t;
			t.cx = dets[di].x + dets[di].w * 0.5f;
			t.cy = dets[di].y + dets[di].h * 0.5f;
			t.w = dets[di].w;
			t.h = dets[di].h;
			t.last_seen = t.last_update = now;
			t.hits = 1;
			tracks_.push_back(t);
		}
	}

	// Позиции на момент `now` с небольшой экстраполяцией.
	void snapshot(double now, std::vector<RenderFace> &out) const
	{
		out.clear();
		for (const Track &t : tracks_) {
			if (t.hits < kMinHits && !t.confirmed)
				continue;
			double since_seen = now - t.last_seen;
			float dt = (float)std::min(0.25, std::max(0.0, now - t.last_update));
			// если лицо давно не видели — не двигаем рамку «вслепую»
			float k = since_seen > 0.25 ? 0.0f : 1.0f;
			RenderFace r;
			r.cx = t.cx + t.vx * dt * k;
			r.cy = t.cy + t.vy * dt * k;
			r.w = t.w;
			r.h = t.h;
			out.push_back(r);
		}
	}

private:
	static constexpr int kMinHits = 2; // отсев одиночных ложных срабатываний

	struct Track {
		float cx = 0, cy = 0, w = 0, h = 0;
		float vx = 0, vy = 0;
		int hits = 0;
		bool confirmed = false;
		double last_seen = 0, last_update = 0;
	};

	static float iou(const Track &t, const FaceBox &d)
	{
		float tx0 = t.cx - t.w * 0.5f, ty0 = t.cy - t.h * 0.5f;
		float tx1 = tx0 + t.w, ty1 = ty0 + t.h;
		float dx1 = d.x + d.w, dy1 = d.y + d.h;
		float iw = std::min(tx1, dx1) - std::max(tx0, d.x);
		float ih = std::min(ty1, dy1) - std::max(ty0, d.y);
		if (iw <= 0 || ih <= 0)
			return 0.0f;
		float inter = iw * ih;
		float uni = t.w * t.h + d.w * d.h - inter;
		return uni > 0 ? inter / uni : 0.0f;
	}

	static float match_score(const Track &t, const FaceBox &d)
	{
		float s = iou(t, d);
		if (s > 0.05f)
			return s;
		// лицо быстро сместилось: рамки не пересекаются, смотрим на расстояние
		float dcx = d.x + d.w * 0.5f, dcy = d.y + d.h * 0.5f;
		float dist = std::hypot(t.cx - dcx, t.cy - dcy);
		float ref = std::max(t.w, d.w) * 1.2f;
		if (dist < ref)
			return 0.001f + 0.04f * (1.0f - dist / ref);
		return 0.0f;
	}

	void correct(Track &t, const FaceBox &d, double now)
	{
		float dt = (float)std::min(0.5, std::max(0.001, now - t.last_update));
		float a = smoothing_;
		float b = a * a / (2.0f - a);
		float dcx = d.x + d.w * 0.5f, dcy = d.y + d.h * 0.5f;

		float px = t.cx + t.vx * dt;
		float py = t.cy + t.vy * dt;
		float rx = dcx - px, ry = dcy - py;

		t.cx = px + a * rx;
		t.cy = py + a * ry;
		t.vx += b * rx / dt;
		t.vy += b * ry / dt;
		// ограничение скорости: не больше 3 экранов в секунду
		t.vx = std::min(3.0f, std::max(-3.0f, t.vx));
		t.vy = std::min(3.0f, std::max(-3.0f, t.vy));

		float sa = std::min(1.0f, a * 1.2f);
		t.w += sa * (d.w - t.w);
		t.h += sa * (d.h - t.h);

		t.last_seen = t.last_update = now;
		t.hits++;
		if (t.hits >= kMinHits)
			t.confirmed = true;
	}

	std::vector<Track> tracks_;
	float smoothing_ = 0.6f;
	double hold_sec_ = 0.5;
	int max_faces_ = 8;
};
