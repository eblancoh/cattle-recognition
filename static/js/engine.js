$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#resultTable').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#resultTable').text('');
        $('#resultTable').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                var data = JSON.parse(data)
                let getKeys = Object.keys(data);
                let row = `<tr><th>Etiqueta</th><th>Intervalo de confianza (%)</th></tr>`
                for (let i = 0; i < getKeys.length; i++) {
                    row += `<tr><td>${getKeys[i]}</td><td>${(data[getKeys[i]] * 100).toFixed(2)}</td></tr>`
                }

                $('.loader').hide();
                $('#resultTable').fadeIn("slow");
                $('#resultTable').append(row);
                // console.log('Success!');
            },
        });
    });

});