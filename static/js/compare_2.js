function copyToClipboard() {
  const pre = document.querySelector('.bibcontainer pre');
  const textToCopy = pre.textContent;
  navigator.clipboard.writeText(textToCopy).then(() => {
    console.log('Text copied to clipboard');
  }).catch(err => {
    console.error('Failed to copy text: ', err);
  });
}
 

const videosGroup3 = [
 "./static/video/GTU_1.mp4",
  "./static/video/GTU_2.mp4"
];
let currentIndex3 = 0;

const videoPlayer3 = document.getElementById('video-player-3');
const dotsContainer3 = document.getElementById('dots-container-3');
const prevButton3 = document.getElementById('prev-button-3');
const nextButton3 = document.getElementById('next-button-3');

function initDots3() {
  for (let i = 0; i < videosGroup3.length; i++) {
    let dot = document.createElement('span');
    dot.classList.add('dot');
    if (i === 0) {
      dot.classList.add('active');
    }
    dot.addEventListener('click', () => changeVideo3(i));
    dotsContainer3.appendChild(dot);
  }
}

function updateDots3() {
  const dots = dotsContainer3.querySelectorAll('.dot');
  dots.forEach((dot, index) => {
    if (index === currentIndex3) {
      dot.classList.add('active');
    } else {
      dot.classList.remove('active');
    }
  });
}

function changeVideo3(index) {
  currentIndex3 = (index + videosGroup3.length) % videosGroup3.length;
  let currentSource = videosGroup3[currentIndex3];

  videoPlayer3.classList.remove('show');

  setTimeout(() => {
    videoPlayer3.src = currentSource;
    videoPlayer3.play();

    videoPlayer3.onloadeddata = () => {
      videoPlayer3.classList.add('show');
      updateDots3();
    };
  }, 500);
}

initDots3();
videoPlayer3.addEventListener('ended', () => changeVideo3(currentIndex3 + 1));

// Optional: Handle previous and next buttons
prevButton3.addEventListener('click', () => changeVideo3(currentIndex3 - 1));
nextButton3.addEventListener('click', () => changeVideo3(currentIndex3 + 1));

changeVideo3(0);