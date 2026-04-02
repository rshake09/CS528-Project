/*
 * CS528 Air Drawing — signal_processing.c
 * pipeline: raw accel+gyro -> complementary filter (pitch/roll)
 *                          -> gravity removal (linear accel)
 *                          -> accel magnitude -> stroke detector
 */

#include "signal_processing.h"
#include <string.h>
#include <math.h>

// internal persistent state for filters and stroke detection
typedef struct {
    float grav_x, grav_y, grav_z;   // LP-filtered gravity estimate
    uint8_t stroke_on_count;
    uint8_t stroke_off_count;
} sp_internal_t;

static sp_internal_t g_int;

// helpers
static inline float vec3_mag(float x, float y, float z)
{
    return sqrtf(x*x + y*y + z*z);
}

// accel-only pitch estimate — atan2 stable across full pitch range
static inline float accel_pitch(float ax, float ay, float az)
{
    return atan2f(-ax, sqrtf(ay*ay + az*az)) * (180.0f / (float)M_PI);
}

// accel-only roll estimate
static inline float accel_roll(float ay, float az)
{
    return atan2f(ay, az) * (180.0f / (float)M_PI);
}

/* sp_init */
void sp_init(sp_state_t *s, float ax_g, float ay_g, float az_g)
{
    memset(s, 0, sizeof(sp_state_t));
    memset(&g_int, 0, sizeof(sp_internal_t));

    // seed pitch/roll from accel so first sample has no jump
    s->pitch = accel_pitch(ax_g, ay_g, az_g);
    s->roll  = accel_roll(ay_g, az_g);

    // seed gravity estimate
    g_int.grav_x = ax_g;
    g_int.grav_y = ay_g;
    g_int.grav_z = az_g;
}

// main signal processing function (call every sample w/fresh imu data)
void sp_update(sp_state_t *s,
               float ax_g, float ay_g, float az_g,
               float gx_dps, float gy_dps, float gz_dps,
               float dt_s)
{
    // 1. store raw values
    s->ax = ax_g;  s->ay = ay_g;  s->az = az_g;
    s->gx = gx_dps; s->gy = gy_dps; s->gz = gz_dps;

    // 2. complementary filter: angle = alpha*(angle + gyro*dt) + (1-alpha)*accel_angle
    //    gyro handles fast motion, accel corrects slow drift
    float accel_p = accel_pitch(ax_g, ay_g, az_g);
    float accel_r = accel_roll(ay_g, az_g);

    s->pitch = SP_COMP_ALPHA * (s->pitch + gx_dps * dt_s)
             + (1.0f - SP_COMP_ALPHA) * accel_p;

    s->roll  = SP_COMP_ALPHA * (s->roll  + gy_dps * dt_s)
             + (1.0f - SP_COMP_ALPHA) * accel_r;

    // 3. gravity removal — LP filter tracks gravity, linear accel = raw - estimate
    float lp = 1.0f - SP_COMP_ALPHA;
    g_int.grav_x = SP_COMP_ALPHA * g_int.grav_x + lp * ax_g;
    g_int.grav_y = SP_COMP_ALPHA * g_int.grav_y + lp * ay_g;
    g_int.grav_z = SP_COMP_ALPHA * g_int.grav_z + lp * az_g;

    s->lax = ax_g - g_int.grav_x;
    s->lay = ay_g - g_int.grav_y;
    s->laz = az_g - g_int.grav_z;

    // 4. accel magnitude
    s->accel_mag = vec3_mag(s->lax, s->lay, s->laz);

    // 5. stroke detection (hysteresis + debounce)
    s->stroke_start = false;
    s->stroke_end   = false;

    if (!s->stroke_active) {
        if (s->accel_mag > SP_STROKE_ON_THRESH) {
            g_int.stroke_on_count++;
            g_int.stroke_off_count = 0;
            if (g_int.stroke_on_count >= SP_STROKE_ON_DEBOUNCE) {
                s->stroke_active = true;
                s->stroke_start  = true;
                g_int.stroke_on_count = 0;
            }
        } else {
            g_int.stroke_on_count = 0;
        }
    } else {
        if (s->accel_mag < SP_STROKE_OFF_THRESH) {
            g_int.stroke_off_count++;
            g_int.stroke_on_count = 0;
            if (g_int.stroke_off_count >= SP_STROKE_OFF_DEBOUNCE) {
                s->stroke_active = false;
                s->stroke_end    = true;
                g_int.stroke_off_count = 0;
            }
        } else {
            g_int.stroke_off_count = 0;
        }
    }

    s->sample_id++;
}