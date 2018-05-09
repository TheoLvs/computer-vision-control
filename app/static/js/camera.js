
// --------------------------------------------------------------------------------------------------------------------------------
// CAMERA PREPARATION
// --------------------------------------------------------------------------------------------------------------------------------




//  https://github.com/tensorflow/tfjs-models/tree/master/posenet


async function setupCamera() {
  const video = document.getElementById('video');
  video.width = maxVideoSize;
  video.height = maxVideoSize;

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




const maxVideoSize = 400;
const canvasSize = 400;
const imageScaleFactor = 0.50;
const flipHorizontal = false;
const outputStride = 32;



/**
 * Feeds an image to posenet to estimate poses - this is where the magic happens.
 * This function loops with a requestAnimationFrame method.
 */
function detectPoseInRealTime(video, net) {
  const canvas = document.getElementById('output');
  const ctx = canvas.getContext('2d');
  canvas.width = canvasSize;
  canvas.height = canvasSize;

  async function poseDetectionFrame() {


    let poses = [];
    var minPoseConfidence= 0.1;
    var minPartConfidence= 0.2;

    // Predict with posenet detector
    const pose = await net.estimateSinglePose(video, imageScaleFactor, flipHorizontal, outputStride);

    // Trigger all the possible actions from a pose
    trigger_actions(pose);

    // Update Canvas
    const scale = canvasSize / video.width;
    draw();
    if (score >= minPoseConfidence) {
      drawKeypoints(keypoints, minPartConfidence, ctx, scale);
      drawSkeleton(keypoints, minPartConfidence, ctx, scale);
    }

    // Loop the capture
    requestAnimationFrame(poseDetectionFrame);

  }

  poseDetectionFrame();

}




// --------------------------------------------------------------------------------------------------------------------------------
// ACTTIONS
// --------------------------------------------------------------------------------------------------------------------------------




function trigger_actions(pose){

  update_scores(pose);
  spotify_control(pose);

}


function update_scores(pose){

    score = pose["score"];
    keypoints = pose["keypoints"];

    nose = keypoints[0].score
    lefthand = keypoints[9].score
    righthand = keypoints[10].score


    $("#output-score").html(score);
    $("#noseposition").html(nose);
    $("#lefthandposition").html(lefthand);
    $("#righthandposition").html(righthand);


}


function spotify_control(pose){

    keypoints = pose["keypoints"];

    nose = keypoints[0].score
    lefthand = keypoints[9].score
    righthand = keypoints[10].score

    if (lefthand > 0.6) {
      play_spotify();
    }

    if (righthand > 0.6){
      pause_spotify();
    }


}



async function play_spotify(){

  spotify_token = $("#spotify-token-input").val();
  console.log("Resuming spotify");

  $.ajax({
      type: 'PUT', // Use POST with X-HTTP-Method-Override or a straight PUT if appropriate.
      dataType: 'json', // Set datatype - affects Accept header
      url: "https://api.spotify.com/v1/me/player/play", // A valid URL
      headers: {"Authorization":"Bearer " + spotify_token}, // X-HTTP-Method-Override set to PUT.
  });

}



async function pause_spotify(){

  spotify_token = $("#spotify-token-input").val();
  console.log("Pausing spotify");
  
  $.ajax({
      type: 'PUT', // Use POST with X-HTTP-Method-Override or a straight PUT if appropriate.
      dataType: 'json', // Set datatype - affects Accept header
      url: "https://api.spotify.com/v1/me/player/pause", // A valid URL
      headers: {"Authorization":"Bearer " + spotify_token}, // X-HTTP-Method-Override set to PUT.
  });

}




// --------------------------------------------------------------------------------------------------------------------------------
// DRAW ON CANVAS
// Credits to Google original code
// --------------------------------------------------------------------------------------------------------------------------------




const color = 'aqua';
const lineWidth = 2;

function toTuple({ y, x }) {
  return [y, x];
}

/**
 * Draws a line on a canvas, i.e. a joint
 */
function drawSegment([ay, ax], [by, bx], color, scale, ctx) {
  ctx.beginPath();
  ctx.moveTo(ax * scale, ay * scale);
  ctx.lineTo(bx * scale, by * scale);
  ctx.lineWidth = lineWidth;
  ctx.strokeStyle = color;
  ctx.stroke();
}

/**
 * Draws a pose skeleton by looking up all adjacent keypoints/joints
 */
function drawSkeleton(keypoints, minConfidence, ctx, scale = 1) {
  const adjacentKeyPoints = posenet.getAdjacentKeyPoints(
    keypoints, minConfidence);

  adjacentKeyPoints.forEach((keypoints) => {
    drawSegment(toTuple(keypoints[0].position),
      toTuple(keypoints[1].position), color, scale, ctx);
  });
}

/**
 * Draw pose keypoints onto a canvas
 */
function drawKeypoints(keypoints, minConfidence, ctx, scale = 1) {
  for (let i = 0; i < keypoints.length; i++) {
    const keypoint = keypoints[i];

    if (keypoint.score < minConfidence) {
      continue;
    }

    const { y, x } = keypoint.position;
    ctx.beginPath();
    ctx.arc(x * scale, y * scale, 3, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
  }
}

/**
 * Draw the bounding box of a pose. For example, for a whole person standing
 * in an image, the bounding box will begin at the nose and extend to one of
 * ankles
 */
function drawBoundingBox(keypoints, ctx) {
  const boundingBox = posenet.getBoundingBox(keypoints);

  ctx.rect(boundingBox.minX, boundingBox.minY,
    boundingBox.maxX - boundingBox.minX, boundingBox.maxY - boundingBox.minY);

  ctx.stroke();
}

/**
 * Converts an arary of pixel data into an ImageData object
 */
async function renderToCanvas(a, ctx) {
  const [height, width] = a.shape;
  const imageData = new ImageData(width, height);

  const data = await a.data();

  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    const k = i * 3;

    imageData.data[j + 0] = data[k + 0];
    imageData.data[j + 1] = data[k + 1];
    imageData.data[j + 2] = data[k + 2];
    imageData.data[j + 3] = 255;
  }

  ctx.putImageData(imageData, 0, 0);
}

/**
 * Draw an image on a canvas
 */
function renderImageToCanvas(image, size, canvas) {
  canvas.width = size[0];
  canvas.height = size[1];
  const ctx = canvas.getContext('2d');

  ctx.drawImage(image, 0, 0);
}






function draw() {



    v = document.getElementById('video');
    canvas = document.getElementById('output');
    canvas.width = video.width;
    canvas.height = video.height;
    context = canvas.getContext('2d');
    w = canvas.width;
    h = canvas.height;

    if(v.paused || v.ended) return false; // if no video, exit here

    context.drawImage(v,0,0,w,h); // draw video feed to canvas
   
}

