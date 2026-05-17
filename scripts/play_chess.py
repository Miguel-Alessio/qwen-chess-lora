import modal

# Define the cloud environment with the required libraries
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers", "torch", "accelerate", "chess")
)

app = modal.App("sah-interactiv")

# Using an A10G GPU which is ideal for a 7B model
@app.cls(gpu="A10G", image=image, timeout=1800)
class ChessPlayer:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("🧠 Încărcăm maestrul de șah Qwen-7B... (durează cam un minut)")
        model_id = "Miguel-Alessio/qwen-7b-chess-master"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Load model directly to GPU in BF16 format for speed
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto"
        )
        print("✅ Model încărcat! Gata de joc.")

    @modal.method()
    def genereaza_mutare(self, istoric_mutari):
        import torch
        
        # Format the history into tokens for the model
        inputs = self.tokenizer(istoric_mutari, return_tensors="pt").to(self.model.device)
        
        # Force greedy decoding (do_sample=False) for strict moves
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the generated text
        raspuns_complet = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the newly generated move
        mutare_noua = raspuns_complet[len(istoric_mutari):].strip().split(" ")[0]
        return mutare_noua

@app.local_entrypoint()
def main():
    import chess
    
    board = chess.Board()
    player = ChessPlayer()
    
    print("\n=========================================")
    print("♟️  BĂTĂLIA CU AI-UL QWEN-7B  ♟️")
    print("=========================================")
    print("Tu joci cu Albele. Introdu mutările în format SAN (ex: e4, Nf3).")
    print("Scrie 'exit' pentru a te da bătut.\n")
    
    istoric = ""
    
    while not board.is_game_over():
        # Human turn
        mutare_om = input("Mutarea ta (Alb): ")
        if mutare_om.lower() == 'exit':
            print("Ai părăsit jocul.")
            break
            
        try:
            # Validate and apply human move
            board.push_san(mutare_om)
            istoric += mutare_om + " "
        except ValueError:
            print("❌ Mutare invalidă sau format greșit! Încearcă din nou (ex: e4, Nf3).")
            continue
            
        print("🤖 AI-ul gândește...")
        
        # AI turn (remotely on Modal)
        mutare_ai = player.genereaza_mutare.remote(istoric)
        
        try:
            # Validate and apply AI move
            board.push_san(mutare_ai)
            istoric += mutare_ai + " "
            print(f"👉 AI-ul a mutat (Negru): {mutare_ai}")
        except ValueError:
            print(f"🤯 AI-ul a halucinat o mutare ilegală ('{mutare_ai}').")
            print("Sistem: Ștergem ultima mutare și te lăsăm să joci tu altceva pentru a-l debloca.")
            istoric = istoric[:-len(mutare_om)-1]
            board.pop()
            continue
            
        # Display the board in console
        print("\n" + str(board) + "\n")
        print("-" * 40)
        
    print("\n🏁 Jocul s-a terminat!")
    print("Rezultat (1-0 Alb, 0-1 Negru, 1/2 Remiză):", board.result())
