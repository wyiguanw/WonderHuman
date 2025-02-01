function copyToClipboard() {
  const pre = document.querySelector('.bibcontainer pre');
  const textToCopy = pre.textContent;
  navigator.clipboard.writeText(textToCopy).then(() => {
    console.log('Text copied to clipboard');
  }).catch(err => {
    console.error('Failed to copy text: ', err);
  });
}
 

const videosGroup5 = [
 "./static/video/animation_1.mp4",
  "./static/video/animation_2.mp4"
];
let currentIndex5 = 0;

const videoPlayer5 = document.getElementById('video-player-5');
const dotsContainer5 = document.getElementById('dots-container-5');
const prevButton5= document.getElementById('prev-button-5');
const nextButton5 = document.getElementById('next-button-5');

function initDots5() {
  for (let i = 0; i < videosGroup5.length; i++) {
    let dot = document.createElement('span');
    dot.classList.add('dot');
    if (i === 0) {
      dot.classList.add('active');
    }
    dot.addEventListener('click', () => changeVideo4(i));
    dotsContainer5.appendChild(dot);
  }
}

function updateDots5() {
  const dots = dotsContainer5.querySelectorAll('.dot');
  dots.forEach((dot, index) => {
    if (index === currentIndex5) {
      dot.classList.add('active');
    } else {
      dot.classList.remove('active');
    }
  });
}

function changeVideo5(index) {
  currentIndex5 = (index + videosGroup5.length) % videosGroup5.length;
  let currentSource = videosGroup5[currentIndex5];

  videoPlayer5.classList.remove('show');

  setTimeout(() => {
    videoPlayer5.src = currentSource;
    videoPlayer5.play();

    videoPlayer5.onloadeddata = () => {
      videoPlayer5.classList.add('show');
      updateDots5();
    };
  }, 500);
}

initDots5();
videoPlayer5.addEventListener('ended', () => changeVideo5(currentIndex5 + 1));

// Optional: Handle previous and next buttons
prevButton5.addEventListener('click', () => changeVideo5(currentIndex5 - 1));
nextButton5.addEventListener('click', () => changeVideo5(currentIndex5 + 1));

changeVideo5(0);