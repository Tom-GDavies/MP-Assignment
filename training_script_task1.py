# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Author: Thomas Davies
# Last Modified: 2025-03-10

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import ultralytics
import torch

def train_model():

    # Check that cuda is enabled
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    # Train model
    model = ultralytics.YOLO("yolo11s.pt")

    model.train(
        data="data.yaml",
        epochs=50,
        batch=8,
        imgsz=960,
        workers=2
    )

    # Evaluate model
    metrics = model.val(split="test")
    print(metrics)