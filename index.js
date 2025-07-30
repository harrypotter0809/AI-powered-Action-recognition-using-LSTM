
function showSection(sectionId) {
    document.querySelectorAll('main > section').forEach(section => {
        section.style.display = 'none';
    });
    document.getElementById(sectionId).style.display = 'block';
}


function showContactForm() {
    showSection('contact-form');
}

async function submitPhoneNumber() {
    const phoneNumber = document.getElementById('phone').value;
    const response = await fetch("/validate_number", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ phoneNumber }),
    });
    const action = await response.json();
    if (action) {
      alert(
        `Phone number ${phoneNumber} submitted. Please choose your preferred message method.`
      );
    } else {
      alert("Please enter a valid phone number.");
    }
}

async function sendMessage(method) {
  const phoneNumber = document.getElementById("phone").value;
  if (!phoneNumber) {
    alert("Please enter a phone number first.");
    return;
  }
  const response = await fetch("/send_message", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ method: method, phoneNumber: phoneNumber }),
  });

  const action = await response.json();
  if (action && action.status === "success") {
    alert(
      `Translation will be sent via ${method} to ${phoneNumber} (simulated)`
    );
    showSection("training");
  } else {
    alert(action.message || "An error occurred.");
  }
}


function getLanguageName(languageCode) {
    const languages = {
        'te': 'Telugu',
        'fhi': 'Hindi',
        'ta': 'Tamil',
        'ml': 'Malayalam',
        'kn': 'Kannada',
        'en': 'English'
    };
    return languages[languageCode] || languageCode;
}