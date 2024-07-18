async function loadModel() {
  const model = await tf.loadLayersModel("/tfjs_model/model.json");
  return model;
}

function preprocessImage(imageElement, index) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  canvas.width = imageElement.width;
  canvas.height = imageElement.height;
  ctx.drawImage(imageElement, 0, 0, canvas.width, canvas.height);

  let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  let data = tf.browser.fromPixels(imageData);

  // Display the original image
  let originalCanvas = document.getElementById("originalCanvas" + index);
  tf.browser.toPixels(data, originalCanvas).then(() => {
    // console.log("Original image displayed.");
  });

  // Convert to grayscale image
  const grayData = tf.image.rgbToGrayscale(data);

  // Binarization processing (thresholding)
  const threshold = 128;
  const binaryData = grayData.greater(tf.scalar(threshold)).toFloat();

  const invertedBinaryData = tf.scalar(1).sub(binaryData);

  let binaryCanvas = document.getElementById("binaryCanvas" + index);
  tf.browser.toPixels(invertedBinaryData, binaryCanvas).then(() => {
    // console.log("Binary image displayed.");
  });

  // Resize the image
  const targetSize = [28, 28];
  data = tf.image.resizeBilinear(invertedBinaryData, targetSize);

  // Add batch dimension
  data = data.expandDims(0);

  return data;
}

async function predict() {
  const model = await loadModel();

  const imageUpload = document.getElementById("imageUpload");
  imageUpload.addEventListener("change", (event) => {
    const file = event.target.files[0];
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => {
        const numChars = 4;
        const canvas = document.getElementById("canvas");
        canvas.width = img.width;
        canvas.height = img.height;

        const segmentedImages = [];
        const charWidth = img.width / numChars;
        for (let i = 0; i < numChars; i++) {
          const charCanvas = document.createElement("canvas");
          charCanvas.width = charWidth;
          charCanvas.height = img.height;
          const charCtx = charCanvas.getContext("2d");
          charCtx.drawImage(
            img,
            i * charWidth,
            0,
            charWidth,
            img.height,
            0,
            0,
            charWidth,
            img.height
          );
          segmentedImages.push(preprocessImage(charCanvas, i));
        }

        // Predict each character
        const startTime = Date.now();
        Promise.all(
          segmentedImages.map((imageData) =>
            model.predict(imageData).argMax(-1).data()
          )
        ).then((predictions) => {
          const captcha = predictions
            .map((p) => String.fromCharCode(p[0] + 48))
            .join("");
          document.getElementById("prediction").innerText = captcha;
          console.log(`Prediction: ${captcha} in ${startTime - Date.now()}ms`);
        });
      };
      img.src = reader.result;
    };
    reader.readAsDataURL(file);
  });
}

document.addEventListener("DOMContentLoaded", () => {
  predict();
});
