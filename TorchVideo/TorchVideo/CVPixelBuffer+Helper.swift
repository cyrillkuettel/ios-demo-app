// Copyright (c) 2020 Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

import Accelerate
import Foundation
import UIKit

extension CVPixelBuffer {
    func normalized(_ width: Int, _ height: Int) -> [Float]? {
        let w = CVPixelBufferGetWidth(self)
        let h = CVPixelBufferGetHeight(self)
        let pixelBufferType = CVPixelBufferGetPixelFormatType(self)
        assert(pixelBufferType == kCVPixelFormatType_32BGRA)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(self)
        let bytesPerPixel = 4
        let croppedImageSize = min(w, h)
        CVPixelBufferLockBaseAddress(self, .readOnly)
        let oriX = w > h ? (w - h) / 2 : 0
        let oriY = h > w ? (h - w) / 2 : 0
        guard let baseAddr = CVPixelBufferGetBaseAddress(self)?.advanced(by: oriY * bytesPerRow + oriX * bytesPerPixel) else {
            return nil
        }
        var inBuff = vImage_Buffer(data: baseAddr, height: UInt(croppedImageSize), width: UInt(croppedImageSize), rowBytes: bytesPerRow)
        guard let dstData = malloc(width * height * bytesPerPixel) else {
            return nil
        }
        var outBuff = vImage_Buffer(data: dstData, height: UInt(height), width: UInt(width), rowBytes: width * bytesPerPixel)
        let err = vImageScale_ARGB8888(&inBuff, &outBuff, nil, vImage_Flags(0))
        CVPixelBufferUnlockBaseAddress(self, .readOnly)
        if err != kvImageNoError {
            free(dstData)
            return nil
        }
        var normalizedBuffer: [Float32] = [Float32](repeating: 0, count: width * height * 3)
        for i in 0 ..< width * height {
            normalizedBuffer[i] = Float32(dstData.load(fromByteOffset: i * 4 + 0, as: UInt8.self)) / 255.0  // R
            normalizedBuffer[width * height + i] = Float32(dstData.load(fromByteOffset: i * 4 + 1, as: UInt8.self)) / 255.0 // G
            normalizedBuffer[width * height * 2 + i] = Float32(dstData.load(fromByteOffset: i * 4 + 2, as: UInt8.self)) / 255.0 // B
        }
        free(dstData)
        return normalizedBuffer
    }
}
