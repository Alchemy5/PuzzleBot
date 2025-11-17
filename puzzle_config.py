"""Shared geometric constants for the puzzle setup."""

table_top_z = -0.05 + 0.025

puzzle_center_x = 0.0
puzzle_center_y = -0.60
puzzle_center_z = table_top_z
puzzle_center = (puzzle_center_x, puzzle_center_y, puzzle_center_z)

puzzle_offset = 0.01

upper_right_translation = (
    puzzle_center_x + puzzle_offset,
    puzzle_center_y + puzzle_offset,
    puzzle_center_z,
)
upper_left_translation = (
    puzzle_center_x - puzzle_offset,
    puzzle_center_y + puzzle_offset,
    puzzle_center_z,
)
lower_left_translation = (
    puzzle_center_x - puzzle_offset,
    puzzle_center_y - puzzle_offset,
    puzzle_center_z,
)
lower_right_translation = (
    puzzle_center_x + puzzle_offset,
    puzzle_center_y - puzzle_offset,
    puzzle_center_z,
)

cross_translation = (
    0.0,
    0.68,
    table_top_z + 0.01,
)

trapezoid_translation = (-0.15, 0.55, table_top_z)
infinity_translation = (-0.15, 0.80, table_top_z)
my_piece_translation = (0.12, 0.52, table_top_z)
rectangle_translation = (0.07, 0.80, table_top_z)

camera_height = 0.45
camera_translation = (
    puzzle_center_x,
    puzzle_center_y - 0.20,
    puzzle_center_z + camera_height,
)

tray_camera_translation = (
    0.10,
    0.40,
    puzzle_center_z + camera_height,
)


tray_translations = {
    "rectangle": rectangle_translation,
    "my_piece": my_piece_translation,
    "trapezoid": trapezoid_translation,
    "infinity": infinity_translation,
    "cross": cross_translation,
}
