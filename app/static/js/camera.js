
async function setupCamera() {
  const video = document.getElementById('video');

  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    const stream = await navigator.mediaDevices.getUserMedia({
      'audio': false,
      'video': {
        facingMode: 'user',
        }
    });
    video.srcObject = stream;

    return new Promise(resolve => {
      video.onloadedmetadata = () => {
        resolve(video);
      };
    });
  } else {
    const errorMessage = "This browser does not support video capture, or this device does not have a camera";
    alert(errorMessage);
    return Promise.reject(errorMessage);
  }
}


async function loadVideo() {
  const video = await setupCamera();
  video.play();

  return video;
}



loadVideo()