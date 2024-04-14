$(document).ready(function() {
    // Function to handle webcam button click
    $("#webcamButton").click(function() {
        $.get("/process_webcam_video", function(data, status) {
            $("#status").html("Webcam video processing initiated.");
        });
    });

    // Function to handle upload button click
    $("#uploadButton").click(function() {
        $.get("/process_uploaded_video", function(data, status) {
            $("#status").html("Uploaded video processing initiated.");
        });
    });
});
