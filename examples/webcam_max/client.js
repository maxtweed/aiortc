let pc = null;

function negotiate() {

  //we are only going to receive from webcam,not send
  // the usual getUserMedia() is not needed since we consume only
  pc.addTransceiver('video', { direction: 'recvonly'});
  pc.addTransceiver('audio', { direction: 'recvonly'});
  /*
  
  */
  return pc.createOffer()
  .then( (offer) => {
    return pc.setLocalDescription(offer);
  })
  .then( () => {
    // wait for ICE gathering to complete
    return new Promise( (resolve) => {
      if (pc.iceGatheringState === 'complete') {
        resolve();
      }
      else { function checkState() {
          if (pc.iceGatheringState === 'complete') {
            pc.removeEventListener('icegatheringstatechange', checkState);
            resolve();
          }
        }
        pc.addEventListener('icegatheringstatechange', checkState);
      }
    });
  })
  .then( () => {
    let offer = pc.localDescription;

    // post offer json to /offer route in the aiortc webcam server
    // wait for a response
    // save the response answer in our peer connection
    return fetch('/offer', {
      body: JSON.stringify( {
        sdp: offer.sdp,
        type: offer.type,
      }),
      headers: {'Content-Type': 'application/json' },
      method: 'POST'
    });
  })
  .then( (response) =>  {
    return response.json();
  })
  .then( (answer) =>  {
    return pc.setRemoteDescription(answer);
  })
  .catch(function (e)  {
    alert(e);
  });
}

function start()
{
  let config = {
    sdpSemantics: 'unified-plan'
  };

  if (document.getElementById('use-stun').checked)
  {
    config.iceServers = [
    {
      urls: ['stun:stun.l.google.com:19302']
    }];
  }

  pc = new RTCPeerConnection(config);

  // connect audio / video
  pc.addEventListener('track', function (evt)
  {
    if (evt.track.kind == 'video') {
      document.getElementById('video').srcObject = evt.streams[0];
    } else {
      document.getElementById('audio').srcObject = evt.streams[0];
    }
  });

  document.getElementById('start').style.display = 'none';
  negotiate();
  document.getElementById('stop').style.display = 'inline-block';
}

function stop()
{
  document.getElementById('stop').style.display = 'none';

  // close peer connection
  setTimeout(function ()
  {
    pc.close();
  }, 500);
}
