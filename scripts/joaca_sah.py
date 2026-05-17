import modal

# Definim mediul din cloud și BĂGĂM MODELUL DIRECT ÎN MEMORIA LUI (Cache)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("transformers", "torch", "accelerate", "chess", "huggingface_hub")
    # Această comandă descarcă modelul o singură dată când se construiește containerul!
 
   .run_commands("huggingface-cli download Miguel-Alessio/qwen-7b-chess-master")
)

app = modal.App("sah-interactiv")

# Folosim un GPU A10G care este perfect pentru un model de 7B
@app.cls(gpu="A10G", image=image, timeout=1800)
class ChessPlayer:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("🧠 Încărcăm maestrul de șah Qwen-7B... (durează cam un minut)")
        model_id = "Miguel-Alessio/qwen-7b-chess-master"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Încărcăm modelul direct pe placa video în format BF16 pentru viteză
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto"
        )
        print("✅ Model încărcat! Gata de joc.")

    @modal.method()
    def genereaza_mutare(self, istoric_mutari):
        import torch
        
        # Transformăm istoricul în limbaj pentru model
        inputs = self.tokenizer(istoric_mutari, return_tensors="pt").to(self.model.device)
        
        # Îi cerem modelului să genereze următoarele caractere
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
               
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decodăm răspunsul
        raspuns_complet = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extragem doar mutarea nouă (ceea ce a generat în plus față de prompt)
        mutare_noua = raspuns_complet[len(istoric_mutari):].strip().split(" ")[0]
        return mutare_noua

@app.local_entrypoint()
def main():
    import chess
    player = ChessPlayer() # Conexiunea la cloud rămâne deschisă aici!
    
    while True: # <--- Adăugăm bucla asta mare care ține jocul viu
        board = chess.Board()
        istoric = ""
        
        print("\n=========================================")
        print("♟️  BĂTĂLIA CU AI-UL QWEN-7B (Meci Nou)  ♟️")
        print("=========================================")
        print("Scrie 'exit' pentru a închide de tot scriptul.\n")
        
        while not board.is_game_over():
            mutare_om = input("Mutarea ta (Alb): ")
            
            if mutare_om.lower() == 'exit':
                print("Ai părăsit jocul definitiv.")
                return # Ieșim din program de tot
                
            try:
                board.push_san(mutare_om)
                istoric += mutare_om + " "
            except ValueError:
                print("❌ Mutare invalidă. Încearcă din nou.")
                continue
                
            print("🤖 AI-ul gândește...")
            mutare_ai = player.genereaza_mutare.remote(istoric)
            
            try:
                board.push_san(mutare_ai)
                istoric += mutare_ai + " "
                print(f"👉 AI-ul a mutat (Negru): {mutare_ai}")
            except ValueError:
                print(f"🤯 AI-ul a halucinat o mutare ilegală ('{mutare_ai}').")
                print("Sistem: Ștergem ultima mutare ca să îl deblocăm.")
                istoric = istoric[:-len(mutare_om)-1]
                board.pop()
                continue
                
            print("\n" + str(board) + "\n")
            print("-" * 40)
            
        print("\n🏁 Jocul s-a terminat! Rezultat:", board.result())
        
        # Meciul s-a terminat, dar noi NU închidem scriptul. Întrebăm de o revanșă:
        raspuns = input("\nVrei să joci din nou? (da/nu): ")
        if raspuns.lower() != 'da':
            break
