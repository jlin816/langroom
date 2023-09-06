![](https://github.com/jlin816/langroom/raw/main/banner.png)

<p align="center" font-weight="bold">
A minimal environment to evaluate embodied question answering in interactive agents.
</p>

In LangRoom, agents must learn to both move and talk. LangRoom contains four objects with randomly generated colors. Agents have a partial view of the environment and receives questions "what color is <object>?". In response, they must seek out the correct object and generate the right answer.

# 💬 Getting Started

Play as a human:
```bash
pip install langroom
# Move with WASD
# Speak tokens 1-10 with number keys 1234567890
python run_gui.py
```

Create an environment instance:
```python
import langroom
env = langroom.LangRoom()
ac = {"move": 0, "talk": 0, "reset": True}
obs = env.step(ac)
```
# 📑 Documentation

## Task Structure

There are four objects with fixed positions and randomized colors. By default, the agent has a partially observed view of 5x5 grid cells and cannot see all the objects at once. The environment generates questions "what color is `<object>`?" and then waits ten timesteps before starting to say "it is `<color>`". Agents answer correctly if they output the correct `<color>` token at the same timestep as the environment. After each question-answer sequence, the colors of the objects are re-randomized.

Three reward variants are implemented, specified by the `task` argument:
- `answer-only`: agent is rewarded only for saying the correct color at the right timestep and penalized a small amount for saying things at other timesteps. Use this reward structure for comparability to the original paper Lin et al. (2023).
- `answer-and-echo`: agent is rewarded for predicting tokens that the environment generates (including silences and questions), with a larger reward for saying the correct color at the right timestep.
- `echo`: agent is rewarded for predicting all tokens the environment generates equally.

## Observation Space
The observation and action space definition follows the [embodied](https://github.com/danijar/embodied/blob/d897527510020eef812a684cbbb87afe05bbd785/embodied/core/base.py#L43) environment interface.
- `image (uint8 (resolution, resolution, 3))`: pixel agent-centric local view
- `text (uint32 ())`: ID of the token at the current timestep
- `log_image (uint8 (resolution, 4 * resolution, 3))`: debugging view with additional information rendered with agent view

Following the `embodied` env interface, these keys are also provided in the observation:
- `reward (float32)`: reward at the current timestep
- `is_first (bool)`: True if this timestep is the first timestep of an episode
- `is_last (bool)`: True if this timestep is the last timestep of an episode (terminated or truncated)
- `is_terminal (bool)`: True if this timestep is the last timestep of an episode (terminated)

## Action Space
LangRoom has a dictionary action space that allows the agent to output actions and tokens (i.e. move and speak) simultaneously at each timestep.
- `move (int32 ())`: ID of the movement action from movement action space `[stay down up right left]`
- `talk (int32 ())`: ID of the generated token

Following the `embodied` env interface, the action space also includes:
- `reset (bool)`: set to True to reset the episode

## Vocabulary Size

By default, the vocabulary size is 15 (the minimal number of tokens to ask and answer questions). To test how agents deal with larger vocabularies (and thus larger action spaces), set the `vocab_size` argument. Additional words in the vocabulary will be filled with dummy tokens.

# 🛠️ Development and Issues

New development and extensions to the environment are welcome! For any questions or issues, please open a GitHub issue.

# Citation
```
@article{lin2023learning,
         title={Learning to Model the World with Language},
         author={Jessy Lin and Yuqing Du and Olivia Watkins and Danijar Hafner and Pieter Abbeel and Dan Klein and Anca Dragan},
         year={2023},
         eprint={2308.01399},
         archivePrefix={arXiv},
}
```
