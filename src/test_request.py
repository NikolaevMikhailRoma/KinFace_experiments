import requests

urls = {
    "image1_url": "https://cdnb.artstation.com/p/assets/images/images/002/646/013/large/guang-yang-a11.jpg?146405541g",
    "image2_url": "https://i.pinimg.com/736x/33/17/88/331788ccaeb05bd954c327b8563a8da1.jpg"
}

response = requests.post("http://localhost:8000/predict", json=urls)
result = response.json()
print(f"Probability of kinship: {result['probability']:.2%}")
print(f"Are related: {result['are_related']}")
print(f"Processing time: {result['processing_time_ms']:.2f}ms")