import gradio as gr
from PIL import Image

from app.qa_pipeline import run_image_pipeline, run_text_pipeline

CONFIDENCE_THRESHOLD = 0.80


def handle_image(image: Image.Image):
    if image is None:
        return "Please upload an image.", "", [], {}

    result = run_image_pipeline(image)

    if result["declined"]:
        label_data = {}
        artist_text = "Could not identify artist"
    else:
        artist_text = f"{result['artist']} ({result['confidence']*100:.1f}% confidence)"
        label_data = {name: conf for name, conf in result["top3"]}

    return artist_text, result["explanation"], result["sample_images"], label_data


def handle_question(question: str):
    if not question.strip():
        return "Please type a question.", "", []

    result = run_text_pipeline(question)
    artist_text = result["artist"] if not result["declined"] else "Artist not found"
    return artist_text, result["explanation"], result["sample_images"]


with gr.Blocks(title="ArtSense AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # ArtSense AI
        **An Intelligent Art History Learning System**
        Upload a painting to identify the artist, or ask a question about any of the 50 historical Western artists in our collection.
        """
    )

    with gr.Tabs():
        # --- Tab 1: Image Pipeline ---
        with gr.Tab("Identify Artwork"):
            gr.Markdown("Upload a painting and ArtSense will identify the artist and explain their work.")
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(type="pil", label="Upload a Painting")
                    identify_btn = gr.Button("Identify Artist", variant="primary")
                with gr.Column(scale=2):
                    artist_output = gr.Textbox(label="Identified Artist", interactive=False)
                    confidence_output = gr.Label(label="Top-3 Confidence", num_top_classes=3)
                    explanation_output = gr.Textbox(label="About This Artist", lines=10, interactive=False)
            gr.Markdown("### Sample Artworks from This Artist")
            gallery_output = gr.Gallery(label="Sample Artworks", columns=5, height=220)

            identify_btn.click(
                fn=handle_image,
                inputs=[image_input],
                outputs=[artist_output, explanation_output, gallery_output, confidence_output],
            )

        # --- Tab 2: Text Q&A Pipeline ---
        with gr.Tab("Ask About an Artist"):
            gr.Markdown(
                "Ask anything about an artist. You can use informal phrasing, partial names, or descriptions.\n\n"
                "*Examples: \"Who painted the Sunflowers?\", \"Tell me about the Cubist Spanish artist\", \"Claud Net\"*"
            )
            with gr.Row():
                with gr.Column(scale=1):
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g. Tell me about the painter of water lilies",
                        lines=3,
                    )
                    ask_btn = gr.Button("Ask", variant="primary")
                with gr.Column(scale=2):
                    qa_artist_output = gr.Textbox(label="Artist", interactive=False)
                    qa_explanation_output = gr.Textbox(label="Answer", lines=10, interactive=False)
            gr.Markdown("### Sample Artworks")
            qa_gallery_output = gr.Gallery(label="Sample Artworks", columns=5, height=220)

            ask_btn.click(
                fn=handle_question,
                inputs=[question_input],
                outputs=[qa_artist_output, qa_explanation_output, qa_gallery_output],
            )
            question_input.submit(
                fn=handle_question,
                inputs=[question_input],
                outputs=[qa_artist_output, qa_explanation_output, qa_gallery_output],
            )

    gr.Markdown(
        "_ArtSense AI covers 50 historical Western artists from the 15th to early 20th century. "
        "Powered by EfficientNet-B3 and Claude AI._"
    )


if __name__ == "__main__":
    demo.launch()
