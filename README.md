# OTFS Wireless Communication Simulator (MATLAB)

## Overview

This project implements a link-level simulation of an Orthogonal Time Frequency Space (OTFS) wireless communication system in MATLAB. The simulator models signal transmission in the delay–Doppler domain and evaluates receiver performance under realistic wireless impairments including multipath propagation and Doppler-induced mobility.

The goal of this project is to study how wireless channel effects distort transmitted symbols and to observe the resulting impact on receiver detection and Bit Error Rate (BER).

---

## System Architecture

The simulator follows a complete physical layer signal chain:

1. Random bit generation and QPSK symbol mapping
2. Delay–Doppler domain grid construction with pilot insertion
3. OTFS modulation (Delay–Doppler → Time–Frequency → Time domain)
4. Multipath fading channel with Doppler shifts
5. Receiver demodulation back to the delay–Doppler domain
6. Symbol detection and BER evaluation
7. Performance visualization

This pipeline emulates practical wireless PHY behavior and demonstrates how time-varying channels affect signal integrity.

---

## Key Features

* End-to-end OTFS modulation and demodulation pipeline
* Delay–Doppler signal representation and visualization
* Multipath channel modeling with configurable delay spread
* Doppler mobility simulation
* Receiver detection and symbol recovery
* BER vs SNR performance analysis
* Constellation and grid-domain visualization tools

---

## Channel Modeling

The simulator incorporates realistic wireless impairments:

* Multiple propagation paths with independent gains
* Delay spread causing inter-symbol interference
* Doppler shifts modeling transmitter/receiver mobility

These effects produce spreading in the delay–Doppler grid, allowing observation of channel-induced distortion.

---

## Performance Evaluation

The system measures Bit Error Rate (BER) under varying Signal-to-Noise Ratio (SNR) conditions. This demonstrates:

* Sensitivity of OTFS signals to channel distortion
* Receiver limitations under severe mobility
* Tradeoffs between channel severity and detection reliability

---

## Visualization Outputs

The simulator provides multiple visualizations to build intuition:

* Transmit constellation
* Delay–Doppler domain grid
* Time–frequency representation
* Channel-distorted DD grid
* Receiver constellation behavior
* BER vs SNR performance curve

---

## Tools & Technologies

* MATLAB
* Digital signal processing
* Wireless channel modeling
* Communication system simulation

---

## Learning Outcomes

This project demonstrates:

* Delay–Doppler domain signal representation
* Effects of multipath and Doppler on wireless signals
* Receiver detection challenges in time-varying channels
* Performance evaluation of wireless PHY systems

---

## How to Run

1. Open MATLAB
2. Load `main_otfs.m`
3. Run the script
4. Observe visualizations and BER results

No external toolboxes are required beyond standard MATLAB signal processing functions.

---

## Applications

Concepts demonstrated in this simulator are relevant to:

* 5G/6G wireless PHY design
* Mobility-aware communication systems
* Software-defined radio prototyping
* Channel modeling and link-level simulation

---

## Author

**Sauraj Kumar Srivastava**
