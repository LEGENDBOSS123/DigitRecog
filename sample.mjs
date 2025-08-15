function oneHotEncode(label, numClasses) {
    const oneHot = new Array(numClasses).fill(0);
    oneHot[label] = 1;
    return oneHot;
}

export default function (data, batchSize){
    let batchX = [];
    let batchY = [];

    for(let i = 0; i < batchSize; i++){
        const randomIndex = Math.floor(Math.random() * data.length);
        batchX.push(data[randomIndex].image);
        batchY.push(oneHotEncode(data[randomIndex].label, 10));
    }

    return [batchX, batchY];
}