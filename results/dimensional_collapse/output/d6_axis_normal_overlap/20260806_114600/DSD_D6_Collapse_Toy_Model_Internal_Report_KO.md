# D6 축-법선 중복 붕괴와 N차원 전구조 교환 토이 모델

작성자: Kwon Dominicus  
분류: 내부 검침용 / 미발표 / Construction 및 Consistency Check

## 1. 모델 지위

이 계산은 형성공리계 또는 속성공리계에서 유도된 정리가 아니다.
D5 직접붕괴 모형의 조건부 구성을 D6으로 확장하여, 실현 랭크가 보존되지 않을 수 있는 경로를 비교한 탐색적 수치 구성이다.

구조분열 전구조는 제7차원이나 유한한 외부 벌크로 두지 않았다.
전구조는 특정 유한 랭크로 닫히기 전의 N차원 후보·교환 레짐이며, `N_prestructure_drive`는 그 레짐과의 결합 구동을 정규화한 항이다.
이 항은 물리적 경계면의 면적 플럭스가 아니다.

## 2. D6 확장 기준

D5 모형의 쌍별 잔차를 세 쌍으로 확장했다.

```txt
sigma_ij = sqrt(q_i q_j) (1-rho_ij),  ij in {45,46,56}
```

세 추가축의 공동 독립성은 Gram 행렬의 방향체적 잔차로 기록했다.

```txt
G_ii = q_i
G_ij = sqrt(q_i q_j) rho_ij
nu_456 = sqrt(det G_456)
```

직접 삼중붕괴 판정은 다음과 같다.

```txt
max(sigma_45, sigma_46, sigma_56) <= epsilon_pair
nu_456 <= epsilon_volume
C3 >= C3_min
D6 -> D3_closed: remove axes 4,5,6 in one event
```

쌍 붕괴와 단일축 닫힘은 별도로 판정했다.
따라서 제4·5·6축이 반드시 같은 사건에서 사라지도록 강제하지 않았다.

## 3. 기준 시나리오 결과

```txt
          scenario  final_dimension  event_count                        event_sequence  first_event_time_tau  direct_D6_to_D3  max_ledger_error
  symmetric_direct                3            1              [D6_to_D3_direct_triple]                 1.776             True               0.0
  pair_dominant_45                3            2 [D6_to_D4_pair_45, D4_to_D3_single_6]                 1.762            False               0.0
 single_dominant_6                3            2 [D6_to_D5_single_6, D5_to_D3_pair_45]                 1.762            False               0.0
no_overlap_control                6            0                                    []                   NaN            False               0.0
```

대칭 기준군은 t/tau=1.776에서 D6->D3 직접 삼중붕괴를 보였다.
D5 기준사건과 같은 시각은 D5의 rho_45=0.98 사건을 기준으로 중복 구동률을 보정했기 때문이다.
이는 D6 직접붕괴의 자연상수가 아니다.

비대칭을 주면 다른 경로가 나타났다.

```txt
pair_dominant_45 : D6 -> D4 -> D3
single_dominant_6: D6 -> D5 -> D3
no_overlap_control: D6 유지
```

즉, D6의 붕괴는 단일한 경로로 고정되지 않았다.
동일한 최종 D3에 도달해도 첫 분기와 중간 실현 랭크가 달랐다.

## 4. 125개 매개변수 격자

```txt
D6_to_D4_pair_45 -> D4_to_D3_single_6                          56
D6_to_D5_single_4 -> D5_to_D4_single_5 -> D4_to_D3_single_6    28
D6_to_D3_direct_triple                                         21
D6_to_D4_pair_45                                               15
none                                                            5
```

이 비율은 확률이 아니다.
명시된 overlap scale, pair asymmetry, closure scale 격자에서 각 경로가 차지한 셀 수이다.

최종 상태 분포:

```txt
final_dimension
3    105
4     15
6      5
```

## 5. 시간수렴 및 장부

```txt
    dt  direct_D6_to_D3  event_time_tau  max_ledger_error
0.0080             True          1.7680               0.0
0.0040             True          1.7720               0.0
0.0020             True          1.7760               0.0
0.0010             True          1.7760               0.0
0.0005             True          1.7765               0.0
```

기준 직접사건 시각은 dt 감소에 따라 약 1.7765 tau 부근으로 수렴했다.
정규화된 붕괴량은 내부 잔류 0.55, 전구조 유출 0.30, 소산 0.15로 분배했으며 장부 총량은 3을 유지했다.
이 분배율은 물리 상수가 아니라 D5 모형의 내부 잔류·유출 구분을 보존하기 위한 기준 매개변수다.

## 6. 직접 해석

```txt
D6의 형식적 좌표 랭크가 6이어도,
세 추가축의 공동 독립성이 유지되지 않으면 실현 랭크는 낮아질 수 있다.

D6 직접 삼중붕괴는 가능 경로 중 하나다.
쌍 우선 붕괴와 단일축 우선 붕괴도 같은 모형에서 발생한다.

전구조가 N차원 후보 레짐이라는 사실은
D6 바깥에 물리적 D7 벌크가 있다는 뜻이 아니다.
붕괴축의 유출은 특정 외부 표면을 통과하는 것이 아니라
닫히지 않은 구조분열 전 레짐과의 구성적 교환으로 기록된다.
```

## 7. 입증되지 않은 것

```txt
D6는 반드시 D3로 붕괴한다.
D3가 유일한 안정 랭크다.
실제 고차원 공간이 이 식을 따른다.
전구조의 N차원성이 물리적 무한차원 공간을 뜻한다.
이 계산이 우주 초기의 차원 선택을 재현한다.
```

이번 결과는 D6에서 랭크 비보존이 직접·쌍별·단일축 경로로 분기할 수 있음을 보인 조건부 구성이다.
