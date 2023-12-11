let imageElement = document.getElementById('imageElement');
let imageInput = document.getElementById('imageInput');
let captionOutput = document.getElementById('captionOutput');
let ortSession;

async function loadONNXModel(modelPath) {
  ortSession = await onnx.InferenceSession.create(modelPath);
}

async function generateCaption() {
  const imageFile = imageInput.files[0];
  if (!imageFile) {
    showError("Please select an image.");
    return;
  }

  try {
    const image = await loadImage(imageFile);
    imageElement.src = URL.createObjectURL(imageFile);

    // Resize image to 384x384
    const resizedImage = resizeImage(image, 384, 384);

    // Preprocess resized image
    const tensorFrame = processImage(resizedImage);

    // Run inference using ONNX Runtime
    const inputs = { 'enc_x': tensorFrame, 'sos_idx': [sos_idx], 'enc_mask': enc_mask, 'fw_dec_mask': fw_dec_mask, 'bw_dec_mask': bw_dec_mask, 'cross_mask': atten_mask };
    const outputs = await ortSession.run(inputs);

    const output_caption = tokens2description(outputs.pred[0].map(Math.round), coco_tokens.idx2word_list, sos_idx, eos_idx);
    showCaption(`Caption: ${output_caption}`);
  } catch (error) {
    showError(`Error: ${error.message}`);
  }
}

function processImage(image) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  const imgSize = 384;
  canvas.width = imgSize;
  canvas.height = imgSize;
  ctx.drawImage(image, 0, 0, imgSize, imgSize);

  const imageData = ctx.getImageData(0, 0, imgSize, imgSize);
  const pixelData = imageData.data;
  const normalizedData = pixelData.map(value => value / 255);
  const tensorFrame = new onnx.Tensor(normalizedData, 'float32', [1, 3, imgSize, imgSize]);

  return tensorFrame;
}

function loadImage(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(new Image(e.target.result));
    reader.onerror = (e) => reject(e);
    reader.readAsDataURL(file);
  });
}

function resizeImage(image, width, height) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = width;
  canvas.height = height;
  ctx.drawImage(image, 0, 0, width, height);
  const resizedImage = new Image();
  resizedImage.src = canvas.toDataURL();
  return resizedImage;
}

function showCaption(caption) {
  captionOutput.textContent = caption;
  captionOutput.style.color = '#333';
}

function showError(errorMsg) {
  captionOutput.textContent = errorMsg;
  captionOutput.style.color = 'red';
}

// Initialize ONNX model
const modelPath = './rf_model.onnx';
loadONNXModel(modelPath);
