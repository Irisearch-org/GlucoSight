Taruh mobilenetv2_food10.pt hasil training di folder ini.

Checkpoint bukan state_dict polos — isinya dict:
  model_state_dict, arch, num_classes, class_to_idx, image_size

class_to_idx ikut tersimpan di dalam .pt, jadi urutan kelas tidak bisa
tertukar saat load ulang.
