import {MediaRecorder, register} from 'extendable-media-recorder';
import {connect} from 'extendable-media-recorder-wav-encoder';

let mediaRecorder = null;
let audioBlobs = [];
let capturedStream = null;

// Register the extendable-media-recorder-wav-encoder
async connect() {
  await register(await connect());
}

// Starts recording audio
function startRecording() {

  return navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
    }
  }).then(stream => {
      audioBlobs = [];
      capturedStream = stream;

      // Use the extended MediaRecorder library
      mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/wav'
      });

      // Add audio blobs while recording 
      mediaRecorder.addEventListener('dataavailable', event => {
        audioBlobs.push(event.data);
      });

      mediaRecorder.start();
  }).catch((e) => {
    console.error(e);
  });

}