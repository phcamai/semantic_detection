import os
import supervision as sv
import cv2
from ultralytics import YOLO
import gradio as gr
import numpy

#HOME = os.getcwd()
# print(HOME)

# Description
title = "<center><strong><font size='8'>Semantic Detection<font></strong></center>"

description_e = """This is a demo of [Semantic Detection Model](https://github.com/phchamai/semantic_detection).
                   Enjoy!
                
              """
examples = [
    ["assets/sample1.jpeg"],
    # ["assets/sample2.jpg"],
]

default_example = examples[0]

css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"

#image = cv2.imread(IMAGE_PATH)

model = YOLO("yolov8m.pt")

def segment_everything(
    image
):
    result = model(image, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)
    #print(detections)

    # model_seg = YOLO("yolov8m-seg.pt")
    # result = model_seg(image, verbose=False)[0]
    # detections_segmentation = sv.Detections.from_ultralytics(result)
    # print(detections_segmentation)
    detections_filtered = detections[detections.confidence > 0.7]

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    labels = []
    labels_count = {}

    for class_id, confidence in zip(detections_filtered.class_id, detections_filtered.confidence):
        class_name = model.model.names[class_id]
        labels_count[class_name] = labels_count.get(class_name, 0) + 1
        labels.append(f"{class_name} ") # {confidence:.2f}
    
    #print(labels_count)
    
    annotated_image = box_annotator.annotate(
        image.copy(),
        detections=detections_filtered,
    )
    annotated_image = label_annotator.annotate(
        annotated_image, detections=detections_filtered, labels=labels
    )
    #sv.plot_image(image=annotated_image, size=(8, 8))
    return annotated_image, str(labels_count)

cond_img_e = gr.Image(label="Input", value=default_example[0], type="pil")
segm_img_e = gr.Image(label="Output Image", interactive=False, type="pil")

with gr.Blocks(css=css, title="Sematic Detection") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            # Title
            gr.Markdown(title)
        
        # Description
        gr.Markdown(description_e)

    with gr.Tab("Everything mode"):
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=0.5):
                cond_img_e.render()
    
            with gr.Column(scale=0.5):
                segm_img_e.render()
    
        # Submit & Clear
        with gr.Row():
            with gr.Column():
                segment_btn_e = gr.Button(
                    "Start", variant="primary"
                )
                clear_btn_e = gr.Button("Clear", variant="secondary")
            with gr.Column():
                out_result = gr.TextArea(label="Output")

        with gr.Row():
            with gr.Column():
                gr.Markdown("Try some of the examples below ⬇️")
                gr.Examples(
                    examples=examples,
                    inputs=[cond_img_e],
                    outputs=[segm_img_e, out_result],
                    fn=segment_everything,
                    cache_examples=True,
                    examples_per_page=4,
                )
                
    segment_btn_e.click(
        segment_everything,
        inputs=[
            cond_img_e,
        ],
        outputs=[segm_img_e, out_result],
    )

    def clear():
        return None, None

    def clear_text():
        return None, None, None

    clear_btn_e.click(clear, outputs=[cond_img_e, segm_img_e])

demo.queue()
demo.launch()