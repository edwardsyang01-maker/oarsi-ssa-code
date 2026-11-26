import unittest

import torch

from augmentation_policy import augment_knee


IDENTITY = {
    "brightness": [1.0, 1.0],
    "contrast": [1.0, 1.0],
    "gamma": [1.0, 1.0],
    "horizontal_flip_probability": 0.0,
    "rotation_degrees": [0.0, 0.0],
    "translation_fraction": [0.0, 0.0],
    "gaussian_blur_probability": 0.0,
}


def devices():
    return [torch.device("cpu")] + ([torch.device("cuda:0")] if torch.cuda.is_available() else [])


def rng_snapshot():
    return (torch.get_rng_state().clone(),
            [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])


def rng_restore(state):
    torch.set_rng_state(state[0])
    if state[1]:
        torch.cuda.set_rng_state_all(state[1])


class AugmentationPolicyTests(unittest.TestCase):
    def test_shape_range_dtype_and_input_unchanged(self):
        for device in devices():
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                with self.subTest(device=device, dtype=dtype):
                    original = torch.linspace(0, 1, 4 * 19 * 27, device=device).reshape(4, 1, 19, 27).to(dtype)
                    image = original.clone().requires_grad_()
                    output = augment_knee(image, {})
                    self.assertEqual(output.shape, image.shape)
                    self.assertEqual(output.dtype, dtype)
                    self.assertEqual(output.device, device)
                    self.assertFalse(output.requires_grad)
                    self.assertNotEqual(output.data_ptr(), image.data_ptr())
                    self.assertTrue(torch.equal(image, original))
                    self.assertTrue(torch.isfinite(output).all().item())
                    self.assertGreaterEqual(output.min().item(), 0)
                    self.assertLessEqual(output.max().item(), 1)

    def test_disabled_returns_original_without_rng_consumption(self):
        for device in devices():
            image = torch.ones((2, 1, 5, 9), device=device)
            before = rng_snapshot()
            output = augment_knee(image, None, enabled=False)
            after = rng_snapshot()
            self.assertIs(output, image)
            self.assertTrue(torch.equal(before[0], after[0]))
            self.assertEqual(len(before[1]), len(after[1]))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(before[1], after[1])))

    def test_rng_restore_reproduces_output_exactly(self):
        for device in devices():
            image = torch.linspace(0, 1, 3 * 13 * 17, device=device).reshape(3, 1, 13, 17)
            torch.manual_seed(723)
            saved = rng_snapshot()
            first = augment_knee(image, {})
            rng_restore(saved)
            second = augment_knee(image, {})
            self.assertTrue(torch.equal(first, second), str(device))

    def test_different_seeds_change_output(self):
        for device in devices():
            image = torch.linspace(0, 1, 3 * 13 * 17, device=device).reshape(3, 1, 13, 17)
            torch.manual_seed(23)
            first = augment_knee(image, {})
            torch.manual_seed(29)
            second = augment_knee(image, {})
            self.assertFalse(torch.equal(first, second), str(device))

    def test_forced_horizontal_flip_is_exact(self):
        for device in devices():
            image = torch.arange(45, device=device, dtype=torch.float32).reshape(1, 1, 5, 9) / 44
            output = augment_knee(image, {**IDENTITY, "horizontal_flip_probability": 1.0})
            self.assertTrue(torch.equal(output, image.flip(-1)), str(device))

    def test_identity_is_exact(self):
        for device in devices():
            image = torch.linspace(0, 1, 45, device=device).reshape(1, 1, 5, 9)
            self.assertTrue(torch.equal(augment_knee(image, IDENTITY), image))

    def test_translation_is_forward_and_border_filled(self):
        for device in devices():
            image = torch.arange(9, device=device, dtype=torch.float32).view(1, 1, 1, 9).expand(1, 1, 5, 9) / 8
            output = augment_knee(image, {**IDENTITY, "translation_fraction": [1 / 9, 1 / 9]})
            expected = torch.cat([image[..., :1], image[..., :-1]], dim=-1)
            torch.testing.assert_close(output, expected, atol=2e-6, rtol=0)

    def test_rotation_respects_non_square_pixel_coordinates(self):
        for device in devices():
            image = torch.zeros((1, 1, 5, 9), device=device)
            image[0, 0, 2, 5] = 1
            output = augment_knee(image, {**IDENTITY, "rotation_degrees": [90, 90]})
            expected = torch.zeros_like(image)
            expected[0, 0, 3, 4] = 1
            torch.testing.assert_close(output, expected, atol=2e-6, rtol=0)

    def test_blur_preserves_constant_and_uses_border_padding(self):
        for device in devices():
            image = torch.full((4, 1, 7, 11), 0.6, device=device)
            output = augment_knee(image, {**IDENTITY, "gaussian_blur_probability": 1.0})
            torch.testing.assert_close(output, image, atol=2e-6, rtol=0)

    def test_rotation_then_translation_uses_inverse_translation(self):
        for device in devices():
            image = torch.zeros((1, 1, 9, 9), device=device)
            image[0, 0, 4, 5] = 1
            parameters = {**IDENTITY, "rotation_degrees": [90, 90],
                          "translation_fraction": [1 / 9, 1 / 9]}
            output = augment_knee(image, parameters)
            expected = torch.zeros_like(image)
            expected[0, 0, 6, 5] = 1
            torch.testing.assert_close(output, expected, atol=2e-6, rtol=0)

    def test_autocast_does_not_change_float32_math(self):
        for device in devices():
            image = torch.linspace(0, 1, 2 * 7 * 11, device=device).reshape(2, 1, 7, 11)
            parameters = {**IDENTITY, "gaussian_blur_probability": 1.0, "gaussian_blur_sigma": [0.7, 0.7]}
            expected = augment_knee(image, parameters)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                output = augment_knee(image, parameters)
            self.assertEqual(output.dtype, torch.float32)
            self.assertTrue(torch.equal(output, expected))

    def test_invalid_parameters_fail(self):
        image = torch.zeros((1, 1, 5, 9))
        for parameters in ({"horizontal_flip_probability": 1.1}, {"gamma": [0, 1]},
                           {"gaussian_blur_kernel": 2}, {"padding": "zeros"},
                           {"align_corners": True}, {"unknown_parameter": 1}):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                augment_knee(image, parameters)
        with self.assertRaises(ValueError):
            augment_knee(torch.zeros((1, 3, 5, 9)), {})
        with self.assertRaises(TypeError):
            augment_knee(torch.zeros((1, 1, 5, 9), dtype=torch.uint8), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
