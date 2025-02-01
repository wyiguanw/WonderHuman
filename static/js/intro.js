function copyToClipboard() {
  const pre = document.querySelector('.bibcontainer pre');
  const textToCopy = pre.textContent;
  navigator.clipboard.writeText(textToCopy).then(() => {
    console.log('Text copied to clipboard');
  }).catch(err => {
    console.error('Failed to copy text: ', err);
  });
}

const videos = [
  "./static/video/WonderHuman.mp4"
  
];
let currentIndex = 0;

const videoPlayer = document.getElementById('video-player');
const prevButton = document.getElementById('prev');
const nextButton = document.getElementById('next');
const dotsContainer = document.getElementById('dots-container');

function initDots() {
  for (let i = 0; i < videos.length; i++) {
    let dot = document.createElement('span');
    dot.classList.add('dot');
    if (i === 0) {
      dot.classList.add('active');
    }
    dot.addEventListener('click', () => changeVideo(i));
    dotsContainer.appendChild(dot);
  }
}

function updateDots() {
  const dots = document.querySelectorAll('.dot');
  dots.forEach((dot, index) => {
    if (index === currentIndex) {
      dot.classList.add('active');
    } else {
      dot.classList.remove('active');
    }
  });
}

function changeVideo(index) {
  currentIndex = (index + videos.length) % videos.length;
  let currentSource = videos[currentIndex];

  videoPlayer.classList.remove('show');

  setTimeout(() => {
    videoPlayer.src = currentSource;
    videoPlayer.play();

    videoPlayer.onloadeddata = () => {
      videoPlayer.classList.add('show');
      updateDots();
    };
  }, 500);
}

initDots();
videoPlayer.addEventListener('ended', () => changeVideo(currentIndex + 1));
prevButton.addEventListener('click', () => changeVideo(currentIndex - 1));
nextButton.addEventListener('click', () => changeVideo(currentIndex + 1));

changeVideo(0);
