// SPDX-License-Identifier: GPL-2.0-or-later
#include <obs-module.h>

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("obs-face-blur", "en-US")

extern void register_face_blur_filter(void);

MODULE_EXPORT const char *obs_module_description(void)
{
	return "Face tracking and blur filter";
}

bool obs_module_load(void)
{
	register_face_blur_filter();
	blog(LOG_INFO, "[obs-face-blur] loaded");
	return true;
}
