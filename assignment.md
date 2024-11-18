# **Multi-Instance Learning (MIL) Lab**

## **Part 1 - MNIST with Multi-Instance Learning**

### **Task 1: Run the MIL_MNIST Example**
Run the provided `MIL_MNIST` example code. This will help you understand how MIL works by applying it to the MNIST dataset, where each bag contains multiple instances (patches).

---

### **Task 2: Modify the Code to Increase the Number of Positive Patches in Each Bag**
- **Objective:** Increase the number of positive instances within each bag.
- **Analysis:** Compare the results of different pooling strategies:
  - **Max pooling**
  - **Mean pooling**

Evaluate how these changes affect the model’s performance. Consider metrics like accuracy, precision, and recall.

---

### **Task 3: Convert the Binary Classification Problem into a Multi-Class Problem**
- **Objective:** Modify the code to handle a **multi-class classification problem**, where bag labels range from 0 to 9.
- **Steps:**
  - Update the dataset to assign multi-class labels to each bag.
  - Modify the loss function and evaluation metrics to support multi-class classification.

## **Part 2 – Camelyon16 Dataset and `mil4wsi` Library**

### **Task 4: Explore the `mil4wsi` Library and Test Available Models**
- **Objective:** Use the provided `mil4wsi_example` to explore different MIL models.
- **Instructions:**
  - Run the code and test various models, including attention-based and simple MIL models.
  - Document the structure and functioning of each model.
  - Compare their performance and key characteristics.

---

### **Task 5: Extract and Visualize Patch Attention**
- **Objective:** Visualize the attention weights at the patch level using the `dsmil` model.
- **Instructions:**
  1. After running a forward pass, the model returns a dictionary:
     ```python
     results = model(data)
     Attentions = results["Higher"][2]
     ```
  2. Extract the spatial coordinates of patches:
     ```python
     data = next(iter(train_loader))
     x, y, x_coord, y_coord = data["data"], data["label"], data["x_coord"], data["y_coord"]
     ```
  3. **Task:** Write a script to generate a **heatmap** by overlaying the attention weights on the corresponding patches. This will help visualize which patches are most relevant for the model’s decision-making.

