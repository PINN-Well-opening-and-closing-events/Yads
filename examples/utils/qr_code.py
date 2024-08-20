import qrcode

# Replace 'https://github.com/username/repository' with your GitHub repository URL
github_repo_url = "https://github.com/PINN-Well-opening-and-closing-events/Yads.git"

# Generate QR code
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)
qr.add_data(github_repo_url)
qr.make(fit=True)

# Create an image from the QR Code instance
img = qr.make_image(fill_color="black", back_color="white")

# Save the image
img.save("github_YADS_repo_qrcode.png")
