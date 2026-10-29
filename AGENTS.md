# TypeSafe integration

For work on TypeSafe or Jev in this project, read and apply the installed
`typesafe-ai` skill at `/Users/josephcai/.codex/skills/typesafe-ai/SKILL.md`.
Use the live documentation at https://docs.typesafe.ai/llms.txt to verify API
schemas and limits when changing the integration.

The gym evaluation uses the exact candidate list supplied by the user.
Do not automatically add exercises or force fixed candidate-set sizes.
The `correct` field is only scored when the supplied ground truth is among
that question's candidates.
