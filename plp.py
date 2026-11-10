import math
from pydub import AudioSegment

# =====================================================================
# 1. PRA-PEMROSESAN SINKRON (DEKODE MP3, TRIM, FRAMING, & HAMMING)
# =====================================================================

def mp3_to_signal_list(file_path, target_sr=16000, threshold=1e-4):
    """Membaca MP3, mengubah ke mono, meresample, dan membuang keheningan awal."""
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1).set_frame_rate(target_sr)
    
    raw_samples = audio.get_array_of_samples()
    s = [sample / 32768.0 for sample in raw_samples]
    
    # Trim keheningan di awal audio (opsional, menjaga kepastian sampel vokal)
    start_idx = next((i for i, x in enumerate(s) if abs(x) > threshold), 0)
    s_trimmed = s[start_idx:]
    
    return s_trimmed, target_sr


def create_frames_as_list(s, sr, frame_length_ms=25, frame_step_ms=10):
    """Memotong sinyal s (list) menjadi list of frames (list of lists)."""
    frame_length = int(sr * (frame_length_ms / 1000.0))
    frame_step = int(sr * (frame_step_ms / 1000.0))
    
    frames = []
    num_samples = len(s)
    
    for start in range(0, num_samples - frame_length + 1, frame_step):
        frame = s[start : start + frame_length]
        frames.append(frame)
        
    return frames


def apply_hamming_window(frame):
    """
    Mengalikan SATU FRAME sinyal audio s(n) dengan Hamming Window w(n).
    Formula: s_windowed(n) = s(n) * (0.54 - 0.46 * cos(2 * pi * n / (N - 1)))
    """
    N = len(frame)
    if N <= 1:
        return frame
        
    windowed_frame = []
    for n in range(N):
        w_n = 0.54 - 0.46 * math.cos((2 * math.pi * n) / (N - 1))
        windowed_frame.append(frame[n] * w_n)
        
    return windowed_frame


# =====================================================================
# 2. KOMPONEN UTAMA METODOLOGI PLP (HERMANSKY, 1990)
# =====================================================================

def compute_power_spectrum(frame):
    """DFT Spektrum Daya P(w) = Re^2 + Im^2"""
    N = len(frame)
    power_spectrum = []

    for k in range(N // 2 + 1):
        real_part = 0.0
        imag_part = 0.0
        for n in range(N):
            angle = 2 * math.pi * k * n / N
            real_part += frame[n] * math.cos(angle)
            imag_part -= frame[n] * math.sin(angle)
            
        p_k = (real_part ** 2) + (imag_part ** 2)
        power_spectrum.append(p_k)
        
    return power_spectrum


def hz_to_bark(f):
    val = f / 600.0
    return 6.0 * math.log(val + math.sqrt(val**2 + 1.0))


def critical_band_filter_weight(delta_bark):
    d = delta_bark
    if d < -1.3:
        return 0.0
    elif -1.3 <= d <= -0.5:
        return 10.0 ** (2.5 * (d + 0.5))
    elif -0.5 < d < 0.5:
        return 1.0
    elif 0.5 <= d <= 2.5:
        return 10.0 ** (-1.0 * (d - 0.5))
    else:
        return 0.0


def critical_band_integration(power_spectrum, sample_rate, num_bark_bands=18):
    N = len(power_spectrum)
    max_freq = sample_rate / 2.0
    max_bark = hz_to_bark(max_freq)
    
    bark_centers = [i * (max_bark / (num_bark_bands - 1)) for i in range(num_bark_bands)]
    freq_bins_bark = [hz_to_bark((k / (N - 1)) * max_freq) for k in range(N)]
        
    theta = []
    for center_bark in bark_centers:
        integrated_val = 0.0
        for k in range(N):
            delta_bark = freq_bins_bark[k] - center_bark
            weight = critical_band_filter_weight(delta_bark)
            integrated_val += power_spectrum[k] * weight
        theta.append(integrated_val)
        
    return theta, bark_centers


def equal_loudness_and_intensity_compression(theta_bark, bark_centers, exponent=0.33):
    phi_spectrum = []
    for theta, bark in zip(theta_bark, bark_centers):
        w2 = bark ** 2
        numerator = (w2 ** 2) * (w2 + 56.4e4)
        denominator = ((w2 + 1.6e5) ** 2) * (w2 + 14.4e6)
        e_weight = numerator / denominator if denominator != 0 else 0.0
        
        xi_val = theta * e_weight
        phi_val = (max(0.0, xi_val)) ** exponent
        phi_spectrum.append(phi_val)
        
    return phi_spectrum


def compute_perceptual_autocorrelation(phi_spectrum, p_order):
    M = len(phi_spectrum)
    R = []
    for k in range(p_order + 1):
        r_k = sum(phi_spectrum[i] * math.cos((math.pi * k * (i + 0.5)) / M) for i in range(M)) / M
        R.append(r_k)
    return R


def levinson_durbin(R, p_order):
    E = R[0]
    a = [0.0] * (p_order + 1)
    
    if E <= 0:
        return [0.0] * p_order, 0.0

    for i in range(1, p_order + 1):
        lambda_val = R[i] + sum(a[j] * R[i - j] for j in range(1, i))
        k_i = -lambda_val / E
        
        a_new = a[:]
        a_new[i] = k_i
        for j in range(1, i):
            a_new[j] = a[j] + k_i * a[i - j]
        a = a_new
        E *= (1.0 - k_i ** 2)
        
    ar_coefficients = a[1:]
    gain = math.sqrt(max(0.0, E))
    return ar_coefficients, gain


def ar_to_cepstral(a_coeffs, gain, num_cepstral=5):
    p = len(a_coeffs)
    c = [0.0] * num_cepstral
    c[0] = math.log(max(gain ** 2, 1e-10))
    
    for n in range(1, num_cepstral):
        sum_val = 0.0
        if n <= p:
            for m in range(1, n):
                sum_val += (1.0 - m / n) * a_coeffs[m - 1] * c[n - m]
            c[n] = -a_coeffs[n - 1] - sum_val
        else:
            for m in range(1, p + 1):
                sum_val += (1.0 - m / n) * a_coeffs[m - 1] * c[n - m]
            c[n] = -sum_val
    return c


# =====================================================================
# 3. PIPELINE UTAMA
# =====================================================================

def extract_plp_features(mp3_file_path, p_order=4, num_cepstral=5):
    # 1. Dekode MP3 ke sinyal list
    s, sr = mp3_to_signal_list(mp3_file_path, target_sr=16000)
    
    # 2. Framing sinyal suara (list of lists)
    frames = create_frames_as_list(s, sr, frame_length_ms=25, frame_step_ms=10)
    
    plp_matrix = []
    
    # 3. Iterasi setiap frame audio
    for raw_frame in frames:
        # A. Hamming Windowing diterapkan per-frame
        windowed_frame = apply_hamming_window(raw_frame)
        
        # B. Power Spectrum via DFT
        power_spec = compute_power_spectrum(windowed_frame)
        
        # C. Critical Band (Bark) Integration
        theta_bark, bark_centers = critical_band_integration(power_spec, sample_rate=sr, num_bark_bands=18)
        
        # D. Equal-Loudness & Intensity Compression
        phi_spec = equal_loudness_and_intensity_compression(theta_bark, bark_centers)
        
        # E. IDFT (Autokorelasi Perseptual R_k)
        R = compute_perceptual_autocorrelation(phi_spec, p_order)
        
        # F. Levinson-Durbin Recursion
        a_coeffs, gain = levinson_durbin(R, p_order)
        
        # G. Konversi ke PLP Cepstral Coefficients
        cepstral_coeffs = ar_to_cepstral(a_coeffs, gain, num_cepstral=num_cepstral)
        
        plp_matrix.append(cepstral_coeffs)
        
    return plp_matrix


# --- EKSEKUSI ---
if __name__ == "__main__":
    mp3_file = "sampel_suara.mp3"
    
    # Jalankan ekstraksi fitur PLP langsung dari MP3
    plp_features = extract_plp_features(mp3_file, p_order=4, num_cepstral=5)
    
    print(f"Total Frame yang Diproses  : {len(plp_features)}")
    print(f"Dimensi Matriks Fitur PLP  : {len(plp_features)} x {len(plp_features[0])}")
    print("\nKoefisien PLP Cepstral Frame Pertama (c_0 s.d c_4):")
    print([round(x, 6) for x in plp_features[0]])
