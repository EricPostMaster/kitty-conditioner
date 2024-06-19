// tfjs_worker.js

importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core');
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-cpu');
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.10');

let model;

// Assuming the prediction indices for cats are known, adjust as needed
const catIndices = [281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293]; // Example indices for different classes of cats

self.onmessage = async function(event) {
    const { modelUrl, imageData } = event.data;
    
    if (modelUrl) {
        tflite.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.10/wasm/');
        try {
            model = await tflite.loadTFLiteModel(modelUrl);
            console.log('Model loaded in worker', model);
            self.postMessage('Model loaded successfully');
        } catch (error) {
            console.error('Error loading model in worker', error);
            self.postMessage(`Error: ${error.message}`);
        }
    }

    if (imageData && model) {
        // Convert imageData (RGBA) to RGB and to uint8
        const rgbData = new Uint8Array(224 * 224 * 3);
        for (let i = 0, j = 0; i < imageData.length; i += 4, j += 3) {
            rgbData[j] = imageData[i];     // R
            rgbData[j + 1] = imageData[i + 1]; // G
            rgbData[j + 2] = imageData[i + 2]; // B
        }

        const tensor = tf.tensor(rgbData, [1, 224, 224, 3], 'int32');
        const predictions = model.predict(tensor);
        
        // Check if a cat is detected
        const isCatDetected = predictions.some(prediction => catIndices.includes(prediction.index));
        // Send back the result to the main thread
        self.postMessage({ isCatDetected });

        const data = predictions.dataSync();
        self.postMessage({ predictions: Array.from(data) });
    }
};
