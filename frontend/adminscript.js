function navigateToDestination() {
    const textarea1Value = document.getElementById("textbox").value;
    const textarea2Value = document.getElementById("timelimit").value;
  
    if (textarea1Value.trim() !== "" && textarea2Value.trim() !== "") {
       const destinationUrl = "candidate.html?textarea1Value=" + encodeURIComponent(textarea1Value) + "&counter=" + encodeURIComponent(textarea2Value);
      
      window.location.href = destinationUrl;
    } else {
      alert("Please fill in both text areas!");
    }

  }

  