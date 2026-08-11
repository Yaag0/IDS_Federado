import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime, timezone

def generate_thesis_pdf_report(results_dir="./results", output_name="reporte_experimentos_FINAL.pdf"):
    pdf_path = os.path.join(results_dir, output_name)
    
    # Recopilar todas las imágenes PNG relevantes en el directorio de trabajo
    imgs = sorted([
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.lower().endswith('.png') and "input_file" not in f # Filtra archivos temporales si los hubiera
    ])
    
    if not imgs:
        print("⚠️ No se encontraron imágenes PNG para generar el reporte.")
        return

    with PdfPages(pdf_path) as pdf:
        # Página de Portada formal
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.5, 0.85, "Informe Oficial de Resultados - QSF-IDS", ha="center", fontsize=20, weight='bold', fontfamily='serif')
        fig.text(0.5, 0.78, "Sistema de Detección de Intrusos Federado y Post-Cuántico", ha="center", fontsize=14, style='italic')
        
        fecha = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        fig.text(0.5, 0.65, f"Fecha de generación (UTC): {fecha}", ha="center", fontsize=10)
        fig.text(0.5, 0.40, "Este documento consolida las métricas de rendimiento, matrices de confusión,\ncurvas ROC y análisis de latencia PQC frente a esquemas tradicionales.", ha="center", fontsize=12)
        pdf.savefig(fig); plt.close()
        
        # Insertar Gráficas dinámicamente
        for img_path in imgs:
            try:
                img = mpimg.imread(img_path)
                fig = plt.figure(figsize=(11.69, 8.27))
                ax = fig.add_subplot(111)
                ax.imshow(img)
                ax.axis('off')
                
                # Título descriptivo basado en el nombre del archivo
                nombre_limpio = os.path.basename(img_path)
                fig.text(0.5, 0.05, f"Figura: {nombre_limpio}", ha="center", fontsize=11, weight='bold')
                pdf.savefig(fig); plt.close()
            except Exception as e:
                print(f"Error procesando {img_path}: {e}")
                
    print(f"📄 Reporte PDF generado exitosamente en: {pdf_path}")

if __name__ == "__main__":
    generate_thesis_pdf_report()