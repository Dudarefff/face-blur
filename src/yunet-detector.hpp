// SPDX-License-Identifier: GPL-2.0-or-later
//
// Детектор лиц YuNet поверх cv::dnn.
//
// В репозитории лежит классическая модель YuNet с тремя выходами
// (loc, conf, iou), поэтому штатный cv::FaceDetectorYN (ему нужна версия
// модели с 12 выходами из OpenCV Zoo) не подходит, и декодирование якорей
// выполняется здесь.
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "face-tracker.hpp"

class YuNetDetector {
public:
	bool load(const std::string &path)
	{
		net_ = cv::dnn::readNetFromONNX(path);
		if (net_.empty())
			return false;
		net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
		net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
		return true;
	}

	// bgr: CV_8UC3. Результат — нормализованные рамки (0..1).
	void detect(const cv::Mat &bgr, float score_thr, float nms_thr, std::vector<FaceBox> &out)
	{
		out.clear();
		const int W = bgr.cols, H = bgr.rows;
		if (W != prior_w_ || H != prior_h_)
			build_priors(W, H);

		// вход: BGR, значения 0..255, без вычитания среднего
		cv::Mat blob = cv::dnn::blobFromImage(bgr, 1.0, cv::Size(W, H), cv::Scalar(), false, false, CV_32F);
		net_.setInput(blob);

		std::vector<cv::Mat> outs;
		net_.forward(outs, out_names_);
		if (outs.size() != 3)
			CV_Error(cv::Error::StsBadSize, "YuNet: unexpected number of outputs");

		const int n = (int)priors_.size();
		cv::Mat loc = outs[0].reshape(1, n);
		cv::Mat conf = outs[1].reshape(1, n);
		cv::Mat iou = outs[2].reshape(1, n);
		if (loc.cols != 14 || conf.cols != 2 || iou.cols != 1)
			CV_Error(cv::Error::StsBadSize, "YuNet: unexpected output shape");

		std::vector<cv::Rect2d> boxes;
		std::vector<float> scores;
		for (int i = 0; i < n; i++) {
			float u = std::min(1.0f, std::max(0.0f, iou.at<float>(i, 0)));
			float p = conf.at<float>(i, 1) * u;
			if (p <= 0.0f)
				continue;
			float s = std::sqrt(p);
			if (s < score_thr)
				continue;

			const float *l = loc.ptr<float>(i);
			const Prior &pr = priors_[(size_t)i];
			float cx = (pr.cx + l[0] * 0.1f * pr.w) * (float)W;
			float cy = (pr.cy + l[1] * 0.1f * pr.h) * (float)H;
			float bw = pr.w * std::exp(l[2] * 0.2f) * (float)W;
			float bh = pr.h * std::exp(l[3] * 0.2f) * (float)H;
			boxes.emplace_back(cx - bw * 0.5f, cy - bh * 0.5f, bw, bh);
			scores.push_back(s);
		}
		if (boxes.empty())
			return;

		std::vector<int> keep;
		cv::dnn::NMSBoxes(boxes, scores, score_thr, nms_thr, keep, 1.0f, 5000);
		for (int k : keep) {
			const cv::Rect2d &b = boxes[(size_t)k];
			out.push_back({(float)(b.x / W), (float)(b.y / H), (float)(b.width / W), (float)(b.height / H)});
		}
	}

private:
	struct Prior {
		float cx, cy, w, h;
	};

	void build_priors(int W, int H)
	{
		static const std::vector<std::vector<int>> kMinSizes = {{10, 16, 24}, {32, 48}, {64, 96}, {128, 192, 256}};
		static const int kSteps[4] = {8, 16, 32, 64};

		// карты признаков на страйдах 8/16/32/64 (целочисленное деление, как в оригинале)
		int fh = (H + 1) / 2 / 2;
		int fw = (W + 1) / 2 / 2;

		priors_.clear();
		for (int k = 0; k < 4; k++) {
			fh /= 2;
			fw /= 2;
			for (int i = 0; i < fh; i++) {
				for (int j = 0; j < fw; j++) {
					for (int ms : kMinSizes[(size_t)k]) {
						priors_.push_back({(j + 0.5f) * (float)kSteps[k] / (float)W,
								   (i + 0.5f) * (float)kSteps[k] / (float)H,
								   (float)ms / (float)W, (float)ms / (float)H});
					}
				}
			}
		}
		prior_w_ = W;
		prior_h_ = H;
	}

	cv::dnn::Net net_;
	std::vector<std::string> out_names_ = {"loc", "conf", "iou"};
	std::vector<Prior> priors_;
	int prior_w_ = 0, prior_h_ = 0;
};
