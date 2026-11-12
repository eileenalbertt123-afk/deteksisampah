import gradio as gr
from detector import detect, CLASSES
from utils import resize_max, draw_detections, count_per_class

def run(image, conf):
    if image is None:
        raise gr.Error("Silakan unggah foto tumpukan sampah dulu.")
    image = resize_max(image)
    dets = detect(image, conf)
    out_img = draw_detections(image, dets)
    summary = count_per_class(dets, CLASSES)
    detail = [[d["label"], d["confidence"], str(d["bbox"])] for d in dets]
    return out_img, summary, detail

with gr.Blocks(title="Deteksi Sampah - Bank Sampah") as demo:
    gr.Markdown("# Deteksi & Klasifikasi Sampah pada Tumpukan\nUnggah foto, lalu sistem menandai tiap objek dan menghitung per kategori.\n\n*Mode demo: model belum terpasang, hasil masih acak.*")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Foto tumpukan sampah")
            conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence threshold")
            btn = gr.Button("Deteksi", variant="primary")
        with gr.Column():
            out = gr.Image(label="Hasil deteksi")
            summary = gr.Dataframe(headers=["Kategori", "Jumlah"], label="Ringkasan")
            detail = gr.Dataframe(headers=["Label", "Confidence", "BBox"], label="Detail")
    with gr.Accordion("Keterbatasan sistem", open=False):
        gr.Markdown(
            "- Hanya 5 kategori: plastik, kertas/kardus, logam, organik, kaca\n"
            "- Tumpukan sangat padat atau tertutup total tidak dijamin terdeteksi\n"
            "- Tidak membedakan sub-jenis plastik (PET, HDPE, PP)\n"
            "- Hanya berbasis citra, tanpa sensor berat atau material\n"
            "- Performa terbaik pada foto terang, fokus, dan sudut wajar\n"
            "- Prototipe, belum diuji langsung di bank sampah"
            )
    btn.click(run, [inp, conf], [out, summary, detail])

if __name__ == "__main__":
    demo.launch()