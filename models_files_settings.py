import os


class ModelsFilesSettings:
    base_dir = "F:/Bureau des émotions/trained_models"

    statedicts_folderpaths_per_style = {
        "surprise": os.path.join(base_dir, "bleu-clair-surprise"),
        "tristesse": os.path.join(base_dir, "bleu-fonce-tristesse"),
        "joie": os.path.join(base_dir, "jaune-joie"),
        "excitation": os.path.join(base_dir, "orange-excitation"),
        "colere": os.path.join(base_dir, "rouge-colere_250-faces"),
        "peur": os.path.join(base_dir, "vert-bleu-peur"),
        "attirance": os.path.join(base_dir, "vert-clair-amour"),
        "degout": os.path.join(base_dir, "violet-degout"),
    }

    iter_number_per_style = {
        "surprise": 50,
        "tristesse": 50,
        "joie": 50,
        "excitation": 52,
        "colere": 50,
        "peur": 50,
        "attirance": 50,
        "degout": 50,
    }
