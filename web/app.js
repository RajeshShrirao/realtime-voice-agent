// Minimal phone browser UI for the voice agent
// Handles WebRTC connection and status updates.

const statusEl = document.getElementById("status");
const statusDot = document.getElementById("statusDot");
const statusLabel = document.getElementById("statusLabel");
const mainBtn = document.getElementById("mainBtn");
const errorMsg = document.getElementById("errorMsg");

let ws = null;
let pc = null;
let callActive = false;

function setStatus(state, label) {
  statusEl.classList.remove("connected", "error", "listening", "speaking");
  if (state) statusEl.classList.add(state);
  statusLabel.textContent = label;
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.classList.add("show");
  setStatus("error", "ERROR");
}

function clearError() {
  errorMsg.classList.remove("show");
  errorMsg.textContent = "";
}

async function startCall() {
  clearError();
  setStatus(null, "CONNECTING");

  try {
    // Create WebSocket connection to signaling server
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${location.host}/ws`;
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      setStatus(null, "SIGNALING...");
    };

    ws.onmessage = async (event) => {
      const data = JSON.parse(event.data);
      const action = data.action;

      if (action === "answer") {
        // We received the answer from server — need to set remote description
        // Actually in this flow, the server creates the answer and sends it back
        // We need to create the offer on the browser side first
        try {
          await createAndSendOffer();
        } catch (e) {
          console.error("Offer error:", e);
          showError("WebRTC negotiation failed");
        }
      } else if (action === "connected") {
        callActive = true;
        setStatus("connected", "CONNECTED");
        mainBtn.textContent = "End Call";
        mainBtn.className = "btn-danger";
        startBargeInMonitoring();
      } else if (action === "error") {
        showError(data.message || "Server error");
      }
    };

    ws.onclose = () => {
      callActive = false;
      setStatus(null, "DISCONNECTED");
      mainBtn.textContent = "Start Call";
      mainBtn.className = "btn-primary";
      stopBargeInMonitoring();
    };

    ws.onerror = () => {
      showError("Connection failed");
    };

  } catch (e) {
    showError("Failed to start: " + e.message);
  }
}

async function createAndSendOffer() {
  // Create local audio track (microphone)
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const audioTrack = stream.getAudioTracks()[0];

  // Create PeerConnection
  pc = new RTCPeerConnection();

  // Add local audio track
  pc.addTrack(audioTrack);

  // Handle incoming remote track (server TTS audio)
  pc.ontrack = (event) => {
    if (event.track.kind === "audio") {
      // Connect remote audio to a media element for playback
      const audioEl = document.createElement("audio");
      audioEl.autoplay = true;
      const dest = new MediaStream();
      dest.addTrack(event.track);
      audioEl.srcObject = dest;
      document.body.appendChild(audioEl);
    }
  };

  // Create offer
  const offer = await pc.createOffer({
    offerToReceiveAudio: true,
    offerToReceiveVideo: false,
  });

  await pc.setLocalDescription(offer);

  // Send offer to server via WebSocket
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      action: "offer",
      sdp: pc.localDescription.sdp,
      type: pc.localDescription.type,
    }));
  }
}

function endCall() {
  if (ws) {
    ws.send(JSON.stringify({ action: "end_call" }));
    ws.close();
  }
  if (pc) {
    pc.close();
    pc = null;
  }
  stopBargeInMonitoring();
  callActive = false;
  setStatus(null, "READY");
  mainBtn.textContent = "Start Call";
  mainBtn.className = "btn-primary";
  // Stop microphone
  navigator.mediaDevices.getUserMedia({ audio: true }).then(s => s.getTracks().forEach(t => t.stop()));
}

// Barge-in monitoring
let bargeInChecker = null;
let bargeInStream = null;

function startBargeInMonitoring() {
  // Already have mic stream from WebRTC — monitor it for barge-in
  navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
    bargeInStream = stream;
    const audioCtx = new AudioContext();
    const source = audioCtx.createMediaStreamSource(stream);
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);

    const dataArray = new Uint8Array(analyser.frequencyBinCount);

    bargeInChecker = setInterval(() => {
      analyser.getByteTimeDomainData(dataArray);
      let sum = 0;
      for (let i = 0; i < dataArray.length; i++) {
        const val = (dataArray[i] - 128) / 128;
        sum += val * val;
      }
      const rms = Math.sqrt(sum / dataArray.length);

      if (rms > 0.05) {
        // User is speaking — send barge-in signal
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ action: "barge_in" }));
        }
        setStatus("listening", "LISTENING");
      }
    }, 100);
  });
}

function stopBargeInMonitoring() {
  if (bargeInChecker) {
    clearInterval(bargeInChecker);
    bargeInChecker = null;
  }
  if (bargeInStream) {
    bargeInStream.getTracks().forEach(t => t.stop());
    bargeInStream = null;
  }
}

// Button handler
mainBtn.addEventListener("click", () => {
  if (callActive) {
    endCall();
  } else {
    startCall();
  }
});

// Initial status
setStatus(null, "READY");
