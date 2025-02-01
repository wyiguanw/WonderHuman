function copyToClipboard() {
  const pre = document.querySelector('.bibcontainer pre');
  const textToCopy = pre.textContent;
  navigator.clipboard.writeText(textToCopy).then(() => {
    console.log('Text copied to clipboard');
  }).catch(err => {
    console.error('Failed to copy text: ', err);
  });
}
 

const videosGroup4 = [
 "./static/video/SIFU_1.mp4",
  "./static/video/ELI_1.mp4"
];
let currentIndex4 = 0;

const videoPlayer4 = document.getElementById('video-player-4');
const dotsContainer4 = document.getElementById('dots-container-4');
const prevButton4= document.getElementById('prev-button-4');
const nextButton4 = document.getElementById('next-button-4');

function initDots4() {
  for (let i = 0; i < videosGroup4.length; i++) {
    let dot = document.createElement('span');
    dot.classList.add('dot');
    if (i === 0) {
      dot.classList.add('active');
    }
    dot.addEventListener('click', () => changeVideo4(i));
    dotsContainer4.appendChild(dot);
  }
}

function updateDots4() {
  const dots = dotsContainer4.querySelectorAll('.dot');
  dots.forEach((dot, index) => {
    if (index === currentIndex4) {
      dot.classList.add('active');
    } else {
      dot.classList.remove('active');
    }
  });
}

function changeVideo4(index) {
  currentIndex4 = (index + videosGroup4.length) % videosGroup4.length;
  let currentSource = videosGroup4[currentIndex4];

  videoPlayer4.classList.remove('show');

  setTimeout(() => {
    videoPlayer4.src = currentSource;
    videoPlayer4.play();

    videoPlayer4.onloadeddata = () => {
      videoPlayer4.classList.add('show');
      updateDots4();
    };
  }, 500);
}

initDots4();
videoPlayer4.addEventListener('ended', () => changeVideo4(currentIndex4 + 1));

// Optional: Handle previous and next buttons
prevButton4.addEventListener('click', () => changeVideo4(currentIndex4 - 1));
nextButton4.addEventListener('click', () => changeVideo4(currentIndex4 + 1));

changeVideo4(0);