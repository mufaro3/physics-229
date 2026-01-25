// https://github.com/adafruit/MAX6675-library/tree/master

#include <float.h>

class MAX6675 {
public:
  MAX6675(int8_t clock_pin, int8_t chip_select_pin, int8_t data_output_pin);
  float read_celsius(void);

private:
  uint8_t  clock_pin, chip_select_pin, data_output_pin;
  uint8_t  read_byte(void);
  uint16_t read_spi(void);
};

/*!
 * @brief Initialize a MAX6675 sensor
 * 
 * @param clock_pin       Serial Clock (SCK) pin
 * @param chip_select_pin Chip Select (CS) pin
 * @param data_output_pin Serial Output (SO) pin
 */
MAX6675::MAX6675(int8_t clock_pin, int8_t chip_select_pin, 
  int8_t data_output_pin) 
{
  this->clock_pin       = clock_pin;
  this->chip_select_pin = chip_select_pin;
  this->data_output_pin = data_output_pin;

  pinMode(chip_select_pin, OUTPUT);
  pinMode(clock_pin,       OUTPUT);
  pinMode(data_output_pin, INPUT);

  digitalWrite(chip_select_pin, HIGH);
}

/*!
 * @brief   Returns the temperature in Celsius
 * @returns Temperature in Kelvin or NAN on error
 */
float MAX6675::read_celsius(void) {
  uint16_t raw_digital_output = read_spi();

  /* No thermocouple attached */
  if (raw_digital_output & 0x4)
    /* return junk data, impossible temperature value */
    return NAN;

  return (float) (raw_digital_output >> 3) / 4.0f;
}

/*! 
 * @brief Reads the next two bytes (full SPI output). 
 * @returns The next 16-bit array or NAN on failure.
 */
uint16_t MAX6675::read_spi(void)
{
  uint16_t raw_digital_output;

  digitalWrite(this->chip_select_pin, LOW);
  delayMicroseconds(10);

  /* read the first two bytes to a 2-byte (16-bit) array */
  raw_digital_output = read_byte();
  raw_digital_output <<= 8;
  raw_digital_output |= read_byte();

  digitalWrite(this->chip_select_pin, HIGH);

  return raw_digital_output;
}

/*! @brief   Reads in the next byte from the output pin.
    @returns The next byte or 0 on an error. */
uint8_t MAX6675::read_byte(void) 
{
  uint8_t data = 0;

  for (int i = 7; i >= 0; i--) 
  {
    digitalWrite(this->clock_pin, LOW);
    delayMicroseconds(10);
    if (digitalRead(this->data_output_pin)) {
      // set the bit to 0 no matter what
      data |= (1 << i);
    }

    digitalWrite(this->clock_pin, HIGH);
    delayMicroseconds(10);
  }

  return data;
}

void setup() {
  Serial.begin(9600);
  delay(500);
}

static const uint8_t CLOCK_PIN   = 7;
static const uint8_t DATA_OUTPUT = 6;

static const uint8_t CHIP_SELECT_1 = 4;
static const uint8_t CHIP_SELECT_2 = 3;
static const uint8_t CHIP_SELECT_3 = 2;

MAX6675 tc_1(CLOCK_PIN, CHIP_SELECT_1, DATA_OUTPUT);
MAX6675 tc_2(CLOCK_PIN, CHIP_SELECT_2, DATA_OUTPUT);
MAX6675 tc_3(CLOCK_PIN, CHIP_SELECT_3, DATA_OUTPUT);

void loop() {
  float temp1, temp2, temp3;
  
  temp1 = tc_1.read_celsius();
  temp2 = tc_2.read_celsius();
  temp3 = tc_3.read_celsius();

  Serial.print(temp1);
  Serial.print(',');
  Serial.print(temp2);
  Serial.print(',');
  Serial.print(temp3);
  Serial.println();
 
  /* For the MAX6675 to update, you must 
   * delay AT LEAST 250ms between reads! */
  delay(1000);
}
