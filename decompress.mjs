/**
 * Asynchronously fetches a gzipped JSON file from a given filename,
 * decompresses it using the native Compression Streams API,
 * and parses it into a JavaScript object.
 * * @param {string} filename The path to the gzipped JSON file.
 * @returns {Promise<any>} A promise that resolves with the parsed JSON data.
 * @throws {Error} Throws an error if the fetch or decompression fails.
 */
async function loadCompressedJson(filename) {
    try {
        // 1. Fetch the gzipped file as a stream.
        const response = await fetch(filename);

        // Check for a successful HTTP response.
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status} for file: ${filename}`);
        }

        // Check if the browser supports the DecompressionStream API.
        if (typeof DecompressionStream === 'undefined') {
            console.error("The DecompressionStream API is not supported in this browser.");
            throw new Error("DecompressionStream API not supported.");
        }

        // 2. Create a decompression stream for Gzip.
        const decompressionStream = new DecompressionStream("gzip");

        // 3. Pipe the compressed response body through the decompression stream.
        const decompressedStream = response.body.pipeThrough(decompressionStream);

        // 4. Read the decompressed stream and decode it as a full string.
        const reader = decompressedStream.getReader();
        const decoder = new TextDecoder("utf-8");
        let fullString = "";

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            fullString += decoder.decode(value, { stream: true });
        }

        // Handle any remaining data in the decoder
        fullString += decoder.decode();

        // 5. Parse the final JSON string and return the object.
        return JSON.parse(fullString);

    } catch (error) {
        // Log and re-throw the error for the calling function to handle.
        console.error("An error occurred while loading compressed JSON:", error);
        throw error;
    }
}

export { loadCompressedJson };