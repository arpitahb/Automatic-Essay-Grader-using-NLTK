
var stopflag = false
document.addEventListener("DOMContentLoaded", function() {

    getprompt();
    document.getElementById('myForm').addEventListener('submit', function(event) {
      
      stopflag=true
      event.preventDefault(); // Prevent form submission
      
      var message = document.getElementById('textbox').value;
      console.log(message);
      // Send the request using fetch
      fetch('http://127.0.0.1:3000', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: message })
      })
      .then(function(response) {
        if (response.ok) {
          return response.json();
        } else {
        
          throw new Error('Error: ' + response.status);
        }
      })
      .then(function(data) {
        console.log(data)
        document.getElementById('response').innerHTML ="Score:" +data.score;
      })
      .catch(function(error) {
        document.getElementById('response').innerHTML = error.message;
      });
    });
  });

function getprompt() {
    const urlParams = new URLSearchParams(window.location.search);
    const textarea1Value = urlParams.get('textarea1Value');
    const textarea2Value = urlParams.get('counter');
    console.log(textarea2Value);
    // document.getElementById("prompt").textContent = decodeURIComponent(textarea1Value);
      let min=parseInt(textarea2Value,10)
      let sec=0
      console.log(min);
    function run(){
      sec-=1;
      if(sec<=0){
        min-=1;
        sec=59;
      }
      document.getElementById("prompt").textContent=`TOPIC : ${textarea1Value} ------------ TIMER: ${min} : ${sec}`;
    }
    if(stopflag!=true){
    setInterval(run,1000);}

  }

  
 