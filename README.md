# Qwen-7B Chess Fine-Tuning ♟️

This repository contains the inference code and technical documentation for the fine-tuning of the **Qwen-7B** model, adapted to generate chess moves in SAN (Standard Algebraic Notation) format. 

The project was developed using the **Surogate** framework and trained on the **Modal** cloud infrastructure.

## 🚀 Model Access (Hugging Face)
Due to its large size (approx. 15 GB), the merged model weights (LoRA + Base) are hosted on the Hugging Face Hub:
👉 **[Miguel-Alessio/qwen-7b-chess-master](https://huggingface.co/Miguel-Alessio/qwen-7b-chess-master)**

## 📊 Training Details & Metrics
The training consisted of processing ~192,000 games, using the LoRA (Low-Rank Adaptation) technique to reduce computational costs while maintaining stability on the 7-billion parameter architecture.

* **Training Type:** SFT (Supervised Fine-Tuning)
* **Estimated Epochs:** ~1
* **Optimization Steps:** 12,000
* **Precision:** BF16
* **Final Training Loss:**
  * **Average:** 0.4329
  * **Median:** 0.4335

The model's convergence (demonstrated in `logs/trainer_state.json`) indicates excellent stabilization of the mathematical text-learning process during the final batches.

## 🔬 Technical Observations and Limitations (Post-Training)
Although the training metrics are very good, empirical testing of the model in real-game conditions highlighted the fundamental limitations of Large Language Models (LLMs) 
compared to symbolic chess engines (e.g., Stockfish):

1. **Lack of spatial reasoning:** Qwen-7B operates on the probabilities of token sequences, not on a 2D representation of the board.
2. **Prompting Sensitivity (Hallucinations):** The model correctly memorized openings (e.g., it responds with `d5` to `e4`), but in the mid-game, it tends to generate illegal 
     moves or formatting artifacts (e.g., generating the number `1.` or moves like `3e5`). This results from a discrepancy between the strict format in the training dataset 
     (which likely included move numbers or ChatML tags) and the raw input format during inference.
3. **Data Volume:** 192,000 games represent an insufficient volume for abstractly deducing game rules from scratch for a 7B parameter model.

## 💻 Interactive Execution
To test the model via the Modal cloud:

1. Install local dependencies:
   `pip install -r requirements.txt`
2. Run the interaction script:
   `modal run scripts/play_chess.py`
## If you encounter an externally-managed-environment error, it is recommended to use a virtual environment (python -m venv .venv). Alternatively, for a quick test, 
   you can bypass the restriction by appending the override flag: pip install -r requirements.txt --break-system-packages
