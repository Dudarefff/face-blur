// Проверка логики трекера без OBS и OpenCV: g++ -std=c++17 -I../src tracker_test.cpp
#include "face-tracker.hpp"
#include <cstdio>
#include <cstdlib>

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { std::printf("FAIL: %s\n", msg); fails++; } else std::printf("ok:   %s\n", msg); } while (0)

int main()
{
	FaceTracker tr;
	tr.set_params(0.6f, 0.5, 8);
	std::vector<RenderFace> out;

	// 1) одиночное ложное срабатывание не должно размываться
	tr.update({{0.4f, 0.4f, 0.1f, 0.15f}}, 0.0);
	tr.snapshot(0.01, out);
	CHECK(out.empty(), "первое обнаружение ещё не подтверждено");

	// 2) второе обнаружение подтверждает лицо
	tr.update({{0.41f, 0.4f, 0.1f, 0.15f}}, 0.033);
	tr.snapshot(0.04, out);
	CHECK(out.size() == 1, "после двух обнаружений лицо подтверждено");

	// 3) лицо движется вправо: трекер должен идти следом, а не отставать
	double t = 0.033;
	float x = 0.41f;
	for (int i = 0; i < 30; i++) {
		t += 1.0 / 30.0;
		x += 0.01f;
		tr.update({{x, 0.4f, 0.1f, 0.15f}}, t);
	}
	tr.snapshot(t + 0.005, out);
	float true_cx = x + 0.05f;
	CHECK(out.size() == 1, "одно лицо в треке (без дублей)");
	CHECK(std::fabs(out[0].cx - true_cx) < 0.02f, "позиция близка к реальной при движении");

	// 4) лицо пропало: рамка держится в пределах hold, потом исчезает
	tr.update({}, t + 0.1);
	tr.snapshot(t + 0.2, out);
	CHECK(out.size() == 1, "рамка держится через 0.2 с после потери");
	tr.update({}, t + 0.7);
	tr.snapshot(t + 0.7, out);
	CHECK(out.empty(), "рамка удалена после таймаута");

	// 5) два лица — два независимых трека
	FaceTracker t2;
	t2.set_params(0.6f, 0.5, 8);
	for (int i = 0; i < 4; i++)
		t2.update({{0.1f, 0.3f, 0.1f, 0.15f}, {0.7f, 0.3f, 0.12f, 0.18f}}, i * 0.033);
	t2.snapshot(0.14, out);
	CHECK(out.size() == 2, "два лица -> два трека");

	// 6) лимит числа лиц
	FaceTracker t3;
	t3.set_params(0.6f, 0.5, 2);
	std::vector<FaceBox> many;
	for (int i = 0; i < 6; i++) many.push_back({0.05f + 0.15f * i, 0.3f, 0.08f + 0.01f * i, 0.12f});
	for (int i = 0; i < 4; i++) t3.update(many, i * 0.033);
	t3.snapshot(0.14, out);
	CHECK(out.size() <= 2, "лимит max_faces соблюдается");

	std::printf(fails ? "\n%d проверок провалено\n" : "\nвсе проверки пройдены\n", fails);
	return fails ? 1 : 0;
}
