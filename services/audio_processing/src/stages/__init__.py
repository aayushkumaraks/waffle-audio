"""Independently replaceable audio-processing stages.

Stage 1 is active today. Stages 2-6 expose narrow contracts so each can be
implemented and benchmarked independently without changing ConversationManager.
"""