# Follow-up confirmation, fixed before generating predictions

The original SFT24 and GRPO6 both collapsed to the majority normal category.
A training-only perception probe correctly recognized all three square colors at
both pixel budgets. SFT24 failed even on training examples. A single training
coverage diagnostic increased SFT updates to 96, retaining seed42, lr1e-4 and
all original training data, LoRA and processor settings. Train/dev reached12/12.
The original test is already viewed and will not be used for further selection.

Freeze this renderer extension now: test24 images,8 source groups, layouts120-127,
SHA256 34511f565c22633a2288005c483e0a2f633d56e1019daac328cf1b5d43dd36c6.
Compare the existing SFT24 and SFT96 checkpoints once, unchanged greedy decoding.
Report strict joint accuracy, each class and all-colors-correct group fraction.
The extension train/dev files are generated solely to preserve generator output;
they are not used for training or selection. No checkpoint selection after seeing
confirmation results. This checks new layouts of the same synthetic grammar,
not generalization to real moderation or a broad distribution shift.
