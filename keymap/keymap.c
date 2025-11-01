#include QMK_KEYBOARD_H
#include "LUFA/Drivers/USB/USB.h"

enum my_layers {
	_HZ2LUT,
	_QWERTY,
	_LOWER,
	_RAISE,
	_ADJUST,
};

enum my_keycodes {
	LANG1 = SAFE_RANGE,
	LANG2,
	L0CK,
};

#define LOWER LT(_LOWER, KC_SPC)
#define RAISE OSL(_RAISE)
#define DOTADJ LT(_ADJUST, KC_DOT)
#define TABADJ LT(_ADJUST, KC_TAB)
#define TMUX C(KC_B)

static bool usb_detached = false;

static void usb_soft_detach(void) {
	cli();
	USB_Detach();
	USB_Disable();
	sei();
	wait_ms(1000);
	usb_detached = true;
}

static void usb_soft_attach(void) {
	cli();
	USB_Init();
	USB_Attach();
	sei();

	uint16_t t = 0;
	while (USB_DeviceState != DEVICE_STATE_Configured && t < 2000) {
		wait_ms(10);
		t += 10;
	}
	if (USB_DeviceState != DEVICE_STATE_Configured) {
		cli();
		USB_Disable();
		USB_Init();
		USB_Attach();
		sei();
	}
	usb_detached = false;
}

const uint16_t PROGMEM keymaps[][MATRIX_ROWS][MATRIX_COLS] = {
	[_HZ2LUT] = LAYOUT(
			KC_F,    KC_U,    KC_N,    KC_G,    _______,                   _______, KC_B,    KC_D,    KC_O,    KC_V,
			KC_L,    KC_I,    KC_S,    KC_T,    _______,                   _______, KC_E,    KC_R,    KC_A,    KC_C,
			KC_W,    KC_Y,    KC_X,    KC_P,    _______, L0CK,    L0CK,    _______, KC_M,    KC_H,    KC_K,    KC_Z,
			OS_LGUI, OS_LALT, OS_LCTL, KC_BSPC, LOWER,   LANG1,   LANG2,   RAISE,   OS_RSFT, OS_RCTL, OS_RALT, OS_RGUI
			),

	[_QWERTY] = LAYOUT(
			KC_Q,    KC_W,    KC_E,    KC_R,    KC_T,                      KC_Y,    KC_U,    KC_I,    KC_O,    KC_P,
			KC_A,    KC_S,    KC_D,    KC_F,    KC_G,                      KC_H,    KC_J,    KC_K,    KC_L,    XXXXXXX,
			KC_Z,    KC_X,    KC_C,    KC_V,    KC_B,    _______, _______, KC_N,    KC_M,    XXXXXXX, XXXXXXX, XXXXXXX,
			_______, _______, _______, _______, _______, _______, _______, _______, _______, _______, _______, _______
			),

	[_LOWER] = LAYOUT(
			_______, KC_BRID, KC_BRIU, KC_MPLY, _______,                   _______, KC_7,    KC_8,    KC_9,    KC_MINS,
			KC_LEFT, KC_DOWN, KC_UP,   KC_RGHT, _______,                   _______, KC_4,    KC_5,    KC_6,    KC_PLUS,
			KC_MRWD, KC_VOLD, KC_VOLU, KC_MFFD, _______, _______, _______, _______, KC_1,    KC_2,    KC_3,    KC_0,
			_______, _______, _______, _______, _______, _______, _______, DOTADJ,  TMUX,    _______, _______, _______
			),

	[_RAISE] = LAYOUT(
			KC_Q,    KC_LCBR, KC_RCBR, KC_COLN, _______,                   _______, KC_HASH, KC_PERC, KC_DQT,  KC_J,
			KC_DOT,  KC_LPRN, KC_RPRN, KC_SCLN, _______,                   _______, KC_ESC,  KC_COMM, KC_UNDS, KC_ENT,
			KC_LT,   KC_PLUS, KC_MINS, KC_GT,   _______, _______, _______, _______, KC_EQL,  KC_SLSH, KC_ASTR, KC_BSLS,
			_______, _______, _______, KC_DEL,  TABADJ,  _______, _______, _______, CW_TOGG, _______, _______, _______
			),

	[_ADJUST] = LAYOUT(
			DM_RSTP, _______, DM_REC2, DM_REC1, _______,                   _______, KC_F7,   KC_F8,   KC_F9,   KC_F12,
			KC_HOME, KC_PGDN, KC_PGUP, KC_END,  _______,                   _______, KC_F4,   KC_F5,   KC_F6,   KC_F11,
			KC_INS,  KC_CAPS, DM_PLY2, DM_PLY1, _______, _______, _______, _______, KC_F1,   KC_F2,   KC_F3,   KC_F10,
			_______, _______, _______, _______, _______, _______, _______, _______, _______, _______, _______, _______
			),
};

bool process_record_user(uint16_t keycode, keyrecord_t *record) {
	if(record->event.pressed) {
		switch(keycode) {
			case LANG1:
				tap_code16(C(KC_SPC));
			case LANG2:
				if (get_highest_layer(default_layer_state) == _HZ2LUT)
					set_single_persistent_default_layer(_QWERTY);
				else
					set_single_persistent_default_layer(_HZ2LUT);
				return false;
			case L0CK:
				if (!usb_detached) usb_soft_detach();
				else               usb_soft_attach();
				return false;
		}
	}
	return true;
}

const uint16_t PROGMEM grv_combo[] = {KC_Q, KC_LCBR, COMBO_END};
const uint16_t PROGMEM exlm_combo[] = {KC_LCBR, KC_RCBR, COMBO_END};
const uint16_t PROGMEM ampr_combo[] = {KC_RCBR, KC_COLN, COMBO_END};
const uint16_t PROGMEM pipe_combo[] = {KC_HASH, KC_PERC, COMBO_END};
const uint16_t PROGMEM ques_combo[] = {KC_PERC, KC_DQT, COMBO_END};
const uint16_t PROGMEM quot_combo[] = {KC_DQT, KC_J, COMBO_END};
const uint16_t PROGMEM lbrc_combo[] = {KC_LCBR, KC_RPRN, COMBO_END};
const uint16_t PROGMEM rbrc_combo[] = {KC_RCBR, KC_LPRN, COMBO_END};
const uint16_t PROGMEM circ_combo[] = {KC_LT, KC_PLUS, COMBO_END};
const uint16_t PROGMEM dlr_combo[] = {KC_ASTR, KC_BSLS, COMBO_END};
const uint16_t PROGMEM at_combo[] = {KC_PERC, KC_UNDS, COMBO_END};
const uint16_t PROGMEM tild_combo[] = {KC_DQT, KC_COMM, COMBO_END};
const uint16_t PROGMEM zero_combo[] = {KC_COMM, KC_UNDS, COMBO_END};
const uint16_t PROGMEM one_combo[] = {KC_COMM, KC_ASTR, COMBO_END};

combo_t key_combos[] = {
    COMBO(grv_combo, KC_GRV),
    COMBO(exlm_combo, KC_EXLM),
    COMBO(ampr_combo, KC_AMPR),
    COMBO(pipe_combo, KC_PIPE),
    COMBO(ques_combo, KC_QUES),
    COMBO(quot_combo, KC_QUOT),
    COMBO(lbrc_combo, KC_LBRC),
    COMBO(rbrc_combo, KC_RBRC),
    COMBO(circ_combo, KC_CIRC),
    COMBO(dlr_combo, KC_DLR),
    COMBO(at_combo, KC_AT),
    COMBO(tild_combo, KC_TILD),
    COMBO(zero_combo, KC_0),
    COMBO(one_combo, KC_1),
};
