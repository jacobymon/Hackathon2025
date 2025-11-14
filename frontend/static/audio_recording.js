// import {MediaRecorder, register} from 'extendable-media-recorder';
// import {connect} from 'extendable-media-recorder-wav-encoder';
// import * as lamejs from '@breezystack/lamejs';

// let mediaRecorder = null;
// let audioBlobs = [];
// let capturedStream = null;

// // Register the extendable-media-recorder-wav-encoder
// async function connect() {
//   await register(await connect());
// }

// // Starts recording audio
// function startRecording() {

//   return navigator.mediaDevices.getUserMedia({
//     audio: {
//       echoCancellation: true,
//     }
//   }).then(stream => {
//       audioBlobs = [];
//       capturedStream = stream;

//       // Use the extended MediaRecorder library
//       mediaRecorder = new MediaRecorder(stream, {
//         mimeType: 'audio/wav'
//       });

//       // Add audio blobs while recording 
//       mediaRecorder.addEventListener('dataavailable', event => {
//         audioBlobs.push(event.data);
//       });

//       mediaRecorder.start();
//   }).catch((e) => {
//     console.error(e);
//   });
// }

// function stopRecording() {
//   return new Promise(resolve => {
//     if (!mediaRecorder) {
//       resolve(null);
//       return;
//     }

//     mediaRecorder.addEventListener('stop', () => {
//       const mimeType = mediaRecorder.mimeType;
//       const audioBlob = new Blob(audioBlobs, { type: mimeType });

//       if (capturedStream) {
//         capturedStream.getTracks().forEach(track => track.stop());
//       }

//       resolve(audioBlob);
//     });
    
//     mediaRecorder.stop();
    
//   });
// }

// function playAudio(audioBlob) {
//   if (audioBlob) {
//     const audio = new Audio();
//     audio.src = URL.createObjectURL(audioBlob);
//     audio.play();
//   }
// }

export function convertWavToMp3(wavBlob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = function () {
      const arrayBuffer = this.result;

      // Create a WAV decoder
      // @ts-expect-error - No idea
      const wavDecoder = lamejs.WavHeader.readHeader(new DataView(arrayBuffer));

      // Get the WAV audio data as an array of samples
      const wavSamples = new Int16Array(arrayBuffer, wavDecoder.dataOffset, wavDecoder.dataLen / 2);

      // Create an MP3 encoder
      const mp3Encoder = new lamejs.Mp3Encoder(wavDecoder.channels, wavDecoder.sampleRate, 128);

      // Encode the WAV samples to MP3
      const mp3Buffer = mp3Encoder.encodeBuffer(wavSamples);

      // Finalize the MP3 encoding
      const mp3Data = mp3Encoder.flush();

      // Combine the MP3 header and data into a new ArrayBuffer
      const mp3BufferWithHeader = new Uint8Array(mp3Buffer.length + mp3Data.length);
      mp3BufferWithHeader.set(mp3Buffer, 0);
      mp3BufferWithHeader.set(mp3Data, mp3Buffer.length);

      // Create a Blob from the ArrayBuffer
      const mp3Blob = new Blob([mp3BufferWithHeader], { type: 'audio/mp3' });

      resolve(mp3Blob);
    };

    reader.onerror = function (error) {
      reject(error);
    };

    // Read the input blob as an ArrayBuffer
    reader.readAsArrayBuffer(wavBlob);
  });
}