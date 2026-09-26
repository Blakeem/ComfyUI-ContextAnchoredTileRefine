"""Can a text-only question sort prompt fragments into things shown and the picture's style?

Style fragments ("photorealistic oil-painting realism", "ethereal") pass the tile verify on
nearly every tile in ab_tile_tags.py, so they would repeat on every tile beside the style
caption. This asks one choice per fragment with no image, over the real CLIP, and prints the
style probability for each so the split can be judged by eye.

    python tests-AB/ab_prompt_fragments.py
"""

import sys
from pathlib import Path

import ab_env

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_tile_tags as tags_run

CRITERIA = {
    "subject": "a thing, person, animal, place or part of the scene that could be pointed at in the picture",
    "style": "the picture's medium, art style, quality, lighting, colour palette, camera, framing or mood",
}
EXTRA_PROMPTS = {
    "tags": "masterpiece, best quality, 1girl, red dress, standing in a wheat field, sunset, "
            "anime style, (detailed eyes:1.2), bokeh, 35mm film, cat on her shoulder",
}


def main():
    ab_env.bootstrap()
    sys.path.insert(0, str(tags_run.TAGGER_ROOT))
    import torch
    from logit_classifier import ChoiceQuestion, SystemOneRequest
    from logit_tagger import tagging

    clip = tags_run.ab_env_load_clip()
    classifier = tagging.build_classifier(clip)
    prompts = {scene: prompt for scene, (_path, prompt) in tags_run.SCENES.items()} | EXTRA_PROMPTS
    with torch.inference_mode():
        for scene, prompt in prompts.items():
            fragments = tags_run.split_prompt(prompt)
            questions = {f"f{n}": ChoiceQuestion(instructions=f'What does the image prompt phrase "{fragment}" describe',
                                                 criteria=CRITERIA) for n, fragment in enumerate(fragments)}
            response, _ = classifier.classify(SystemOneRequest(state="", questions=questions))
            print(f"== {scene}")
            for n, fragment in enumerate(fragments):
                answer = response.answers[f"f{n}"]
                print(f"  style {answer.probabilities['style']:.3f}  {fragment}")


if __name__ == "__main__":
    main()
