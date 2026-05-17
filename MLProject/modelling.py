import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

def train_baseline_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(script_dir, "ibm_attrition_preprocessing.csv")
    TARGET_COLUMN = "Attrition"

    if not os.path.exists(DATA_PATH):
        print(f"⚠️ Error: Berkas {DATA_PATH} tidak ditemukan!")
        return

    print(f"• Memuat dataset bersih dari: {DATA_PATH}")
    df_clean = pd.read_csv(DATA_PATH)

    # Memisahkan Fitur (X) dan Target (y)
    if TARGET_COLUMN not in df_clean.columns:
        TARGET_COLUMN = df_clean.columns[-1]
        
    X = df_clean.drop(columns=[TARGET_COLUMN])
    y = df_clean[TARGET_COLUMN]
    
    # Splitting data dengan stratify
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Kunci Utama: Jangan pernah set_experiment() atau set_tracking_uri() manual di dalam kode 
    # agar tidak tabrakan dengan parameter yang dikirim oleh CLI/GitHub Actions!
    
    # Mengaktifkan MLflow Autolog
    mlflow.sklearn.autolog()
    print("• MLflow Autolog berhasil diaktifkan.")
    
    # Memulai pencatatan run (with kosong seperti milik temenmu agar otomatis nempel ke CLI)
    with mlflow.start_run() as run:
        print(f"▶ Memulai tracking Run ID: {run.info.run_id}")
        
        # Menggunakan model baseline RandomForest
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        print("• Melatih model baseline RandomForestClassifier...")
        model.fit(X_train, y_train)
        
        # Prediksi dan Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print("\n=== HASIL EVALUASI MODEL BASELINE ===")
        print(f"Accuracy Score: {acc:.4f}")
        print("\n[Classification Report]")
        print(classification_report(y_test, y_pred))
        
        # --- 🔒 KUNCI REVISI REVIEWER DICODING (LOG ARTIFACT) ---
        # Perintah ini memaksa MLflow mencetak berkas MLmodel, conda.yaml, model.pkl ke folder mlruns gess!
        print("• Merekam berkas fisik artifak ke dalam folder tracking mlruns...")
        mlflow.sklearn.log_model(model, artifact_path="saved_model")
        
        # Simpan model secara lokal untuk kebutuhan build-docker nanti gess
        import shutil
        saved_model_path = os.path.join(script_dir, "saved_model")
        shutil.rmtree(saved_model_path, ignore_errors=True)
        mlflow.sklearn.save_model(model, saved_model_path)
        print(f"✓ Sukses menyimpan model lokal di: {saved_model_path}")

if __name__ == "__main__":
    train_baseline_model()