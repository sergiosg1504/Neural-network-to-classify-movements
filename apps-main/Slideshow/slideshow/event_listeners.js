let socket = new WebSocket("ws://localhost:8000/events");

socket.onmessage = function(event) {
  console.log(event.data)
  switch(event.data){
    case "rotate":
      console.log("received 'rotate' event");
      currentSlide = Reveal.getCurrentSlide();
      rotateRotatables(currentSlide, 90);  // defined in helper_methods.js
      break;
    case "swipe_right":
      console.log("received 'swipe_right' event");
      Reveal.right();
      break;
    case "swipe_left":
      console.log("received 'swipe_left' event");
      Reveal.left();
      break;
    case "rotate_left":
      console.log("received 'rotate_left' event");
      currentSlide = Reveal.getCurrentSlide();
      rotateRotatables(currentSlide, -90);  // defined in helper_methods.js
      break;  
    case "zoom_in":
      console.log("received 'zoom_in' event");
      zoom(10); // `zoom()` is defined in helper_methods.js
      break;
    case "zoom_out":
      console.log("received 'zoom_out' event");
      zoom(-10); // `zoom()` is defined in helper_methods.js
      break;
    case "table_flip":
      console.log("received 'rotate_left' event");
      currentSlide = Reveal.getCurrentSlide();
      rotateRotatables(currentSlide, 360);  // defined in helper_methods.js
      break;  
    default:
      console.debug(`unknown message received from server: ${event.data}`);
  }
};
