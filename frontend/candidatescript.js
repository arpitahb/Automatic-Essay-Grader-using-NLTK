document.addEventListener("DOMContentLoaded", function() {
    getprompt();
  });

function getprompt() {
    const urlParams = new URLSearchParams(window.location.search);
    const textarea1Value = urlParams.get('textarea1Value');
    document.getElementById("prompt").textContent = decodeURIComponent(textarea1Value);
  }

  