import NN from "./NN.mjs";
import { loadCompressedJson } from "./decompress.mjs";
import parseData from "./parseData.mjs";
import sample from "./sample.mjs";

const canvas = document.getElementById("draw-canvas");
const ctx = canvas.getContext("2d");
const clearBtn = document.getElementById("clear-btn");
const predictBtn = document.getElementById("predict-btn");
const predictionEl = document.getElementById("prediction");

const nn = new NN([784, 64, 64, 10]);

predictionEl.innerHTML = `Loading 16MB of Data...`;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const data = await loadCompressedJson("mnist_handwritten_train.json.gz");
top.data = data;
parseData(data);
predictionEl.innerHTML = `Data Loaded. Training...`;
await sleep(200)

let epochs = 100;
for (let i = 0; i < epochs; i++) {
    let [batchX, batchY] = sample(data, 256);
    let loss = nn.train(batchX, batchY, 10, 0.0015);
    predictionEl.innerHTML = `Loss: ${loss.toFixed(4)}, Epoch: ${i + 1}/${epochs}`;
    await sleep(1);
}



// main.mjs

let drawing = false;

// Setup drawing style
ctx.lineWidth = 20;
ctx.lineCap = "round";
ctx.strokeStyle = "black";

// Mouse events
canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", stopDraw);
canvas.addEventListener("mouseleave", stopDraw);

// Touch events (mobile)
canvas.addEventListener("touchstart", (e) => startDraw(e.touches[0]));
canvas.addEventListener("touchmove", (e) => {
    draw(e.touches[0]);
    e.preventDefault();
});
canvas.addEventListener("touchend", stopDraw);

function startDraw(e) {
    drawing = true;
    ctx.beginPath();
    ctx.moveTo(getX(e), getY(e));
}

function draw(e) {
    if (!drawing) return;
    ctx.lineTo(getX(e), getY(e));
    ctx.stroke();
}

function stopDraw() {
    drawing = false;
    ctx.closePath();
}

function getX(e) {
    return e.clientX - canvas.getBoundingClientRect().left;
}

function getY(e) {
    return e.clientY - canvas.getBoundingClientRect().top;
}

// Clear button
clearBtn.addEventListener("click", () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    predictionEl.innerHTML = "Prediction: <strong>None</strong>";
});

// Predict button
predictBtn.addEventListener("click", async () => {
    // Resize to 28x28 grayscale
    const smallCanvas = document.createElement("canvas");
    smallCanvas.width = 28;
    smallCanvas.height = 28;
    const smallCtx = smallCanvas.getContext("2d");

    // Fill white background before scaling
    smallCtx.fillStyle = "white";
    smallCtx.fillRect(0, 0, 28, 28);

    // Draw scaled image
    smallCtx.drawImage(canvas, 0, 0, 28, 28);

    // Get image data
    const imgData = smallCtx.getImageData(0, 0, 28, 28).data;

    // Convert RGBA to normalized grayscale
    const inputArray = new Float32Array(28 * 28);
    for (let i = 0; i < 28 * 28; i++) {
        const r = imgData[i * 4];
        const g = imgData[i * 4 + 1];
        const b = imgData[i * 4 + 2];
        // Normalize to 0-1, invert so black lines are 1 and white is 0
        inputArray[i] = 1 - (r + g + b) / (3 * 255);
    }

    // Pass through neural network
    const output = nn.feedForward(inputArray);

    // Find the digit with the highest probability
    let maxIndex = 0;
    let maxVal = -Infinity;
    for (let i = 0; i < output.length; i++) {
        if (output[i] > maxVal) {
            maxVal = output[i];
            maxIndex = i;
        }
    }

    predictionEl.innerHTML = `Prediction: <strong>${maxIndex}</strong> (Confidence: ${(maxVal * 100).toFixed(1)}%) <br> all values: ${Array.from(output).map(v => v.toFixed(3)).join(" | ")}`;
});
